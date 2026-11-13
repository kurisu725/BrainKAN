"""ADHD: Full90 development search, unchanged parameter transfer to training-only EN75."""
from pathlib import Path
import sys, os, time, json, subprocess, gc, traceback, hashlib
import numpy as np
import torch
from storage import read_json, write_json, digest
from runtime_lock import acquire
from trainer import train_candidate, SafePause
from data import load_data, validate_split
from models import build_model
from run_review import input_features, predict
from study import (ROOT, metrics, parameter_count, geometry, write_predictions, read_predictions,
                   verify_files, commit_leader, leader_path, choice_key, retire_checkpoint)

def status(stage, **kw):
    write_json(ROOT / 'status.json', dict(stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%dT%H:%M:%S'), **kw))

def paused(): return (ROOT / 'PAUSE').exists()
def check_pause():
    if paused(): raise SafePause()

def wait_for_gpu(plan):
    dependency = plan['gpu_queue_after'].replace("'", "''")
    command = ("@((Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\") | "
               "Where-Object { $_.CommandLine -and $_.CommandLine.Contains('" + dependency + "') "
               "-and $_.CommandLine.Contains('experiment.py') } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress")
    while True:
        check_pause()
        result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True,
                                text=True, check=True, timeout=30)
        workers = json.loads(result.stdout.strip() or '[]')
        if not workers:
            write_json(ROOT / 'audit/gpu_queue_release.json', dict(dependency=plan['gpu_queue_after'],
                       worker_processes=[], time=time.strftime('%Y-%m-%dT%H:%M:%S')))
            return
        if isinstance(workers, dict): workers = [workers]
        status('waiting_gpu', dependency=plan['gpu_queue_after'], worker_pids=[p['ProcessId'] for p in workers])
        time.sleep(15)

def reuse(plan, row, split, y):
    source = Path(plan['old_study']) / 'runs' / f"{row['branch']}_p{split['id']}"
    old = read_json(source / 'complete.json')
    assert old['config'] == row['config'] and old['rois'] == row['rois'] and not old['final_test_used']
    verify_files(old['files'], source)
    ids, labels, prob = read_predictions(source / 'development_predictions.csv')
    assert ids.tolist() == split['indices']['valid'] and np.array_equal(labels, y[ids])
    m = metrics(labels, prob)
    for key, value in m.items(): assert abs(value - old['development'][key]) < 1e-7
    history = read_json(source / 'history.json')
    assert len(history) == row['config']['epochs']
    ck = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
    assert ck['config'] == row['config'] and ck['epoch'] == row['config']['epochs'] and ck['n_roi'] == row['n_roi']
    del ck
    row.update(development=m, reused=True, source=str(source), source_model_sha256=digest(source / 'model.pt'))
    return prob, history, source / 'model.pt'

def run_job(plan, phash, split, candidate, branch, rois, x, y):
    p = split['id']; cfg = candidate['config']; job = f"{branch}_p{p}_{candidate['id']}"
    folder = ROOT / 'runs' / job; folder.mkdir(exist_ok=True)
    marker = folder / 'complete.json'; idx = split['indices']
    if marker.exists():
        row = read_json(marker)
        assert row['config'] == cfg and row['rois'] == rois and row['plan_sha256'] == phash
        verify_files(row['files'], folder)
        return row
    check_pause(); done = len(list((ROOT / 'runs').glob('*/complete.json')))
    status('developing', completed=done, total=65, job=job, epoch=0)
    row = dict(job=job, branch=branch, partition=p, candidate=candidate['id'], config=cfg, rois=rois,
               n_roi=len(rois), parameter_count=parameter_count(len(rois), cfg), plan_sha256=phash, final_test_used=False)
    payload = None; external = None
    if candidate['id'] == plan['baseline_id']:
        prob, history, external = reuse(plan, row, split, y)
    else:
        def on_epoch(epoch, entry, saved):
            status('developing', completed=done, total=65, job=job, epoch=epoch, epochs=cfg['epochs'],
                   saved_epoch=saved, development=entry['development'], epoch_seconds=entry['seconds'])
            print(job, 'epoch', epoch, 'train_ce', round(entry['training_ce'], 6),
                  'dev_acc', round(entry['development']['accuracy'], 6), flush=True)
        tr, dv = idx['train'], idx['valid']
        payload, prob, history = train_candidate(x[tr][:, rois][:, :, rois], y[tr], x[dv][:, rois][:, :, rois], y[dv],
            cfg, job, rois, phash, ROOT, on_epoch=on_epoch, pause_check=paused, interval=plan['checkpoint_interval'])
        row.update(development=metrics(y[dv], prob), reused=False)
        h = hashlib.sha256()
        for k, v in payload['state'].items(): h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
        row['state_tensor_sha256'] = h.hexdigest()
    write_json(folder / 'config.json', row); write_json(folder / 'history.json', history)
    write_predictions(folder / 'development_predictions.csv', idx['valid'], y[idx['valid']], prob)
    row['files'] = {name: digest(folder / name) for name in ['config.json', 'history.json', 'development_predictions.csv']}
    commit_leader(ROOT, row, payload=payload, external=external)
    write_json(marker, row); retire_checkpoint(ROOT, job)
    del payload; gc.collect(); torch.cuda.empty_cache()
    return row

def main():
    lock = acquire(ROOT / 'state/worker.lock')
    if lock is None: return
    try:
        plan = read_json(ROOT / 'plan.json'); phash = digest(ROOT / 'plan.json')
        assert phash == read_json(ROOT / 'plan_seal.json')['sha256']
        verify_files(plan['source_sha256'], ROOT); verify_files(plan['previous_artifacts'])
        assert digest(plan['dataset']['path']) == plan['dataset']['sha256']
        check_pause(); wait_for_gpu(plan); torch.set_num_threads(4)
        x, y, _, ph = load_data(plan['dataset']['path'], plan['dataset']['label_column'])
        assert len(y) == 847 and len(np.unique(ph[:, plan['dataset']['subject_column']])) == 847
        if (ROOT / 'state/current.pt').exists():
            checkpoint = torch.load(ROOT / 'state/current.pt', map_location='cpu', weights_only=False)
            job = checkpoint['job']; del checkpoint
            if (ROOT / 'runs' / job / 'complete.json').exists(): retire_checkpoint(ROOT, job)
        rows = []; full_choices = []
        for split in plan['splits']:
            validate_split(split['indices'], len(y), y)
            group = [run_job(plan, phash, split, c, 'full90', list(range(90)), x, y) for c in plan['candidates']]
            rows.extend(group)
            chosen = read_json(leader_path(ROOT, 'full90', split['id']))
            assert chosen['record']['job'] == min(group, key=choice_key)['job']
            assert digest(chosen['model_path']) == chosen['model_sha256']
            full_choices.append(chosen)
        choice_record = dict(plan_sha256=phash, choices=full_choices, final_test_used=False)
        if (ROOT / 'parameter_choices.json').exists(): assert read_json(ROOT / 'parameter_choices.json') == choice_record
        else: write_json(ROOT / 'parameter_choices.json', choice_record)
        en_choices = []
        for split, full in zip(plan['splits'], full_choices):
            selection = read_json(ROOT / 'selections' / f"p{split['id']}.json")
            assert selection['fit_indices'] == split['indices']['train'] and len(selection['rois']) == 75
            c = dict(id=full['record']['candidate'], config=full['record']['config'])
            row = run_job(plan, phash, split, c, 'en75', selection['rois'], x, y); rows.append(row)
            chosen = read_json(leader_path(ROOT, 'en75', split['id']))
            assert chosen['record']['job'] == row['job']; en_choices.append(chosen)
        assert len(rows) == 65
        choices = full_choices + en_choices
        files = {str(p.relative_to(ROOT)): digest(p) for d in ['runs', 'selections'] for p in (ROOT / d).rglob('*') if p.is_file()}
        files['parameter_choices.json'] = digest(ROOT / 'parameter_choices.json')
        for c in choices:
            files[c['model_path']] = c['model_sha256']
            path = leader_path(ROOT, c['record']['branch'], c['record']['partition'])
            files[str(path.relative_to(ROOT))] = digest(path)
        frozen = dict(plan_sha256=phash, records=65, choices=choices, files=files, final_test_used=False)
        if (ROOT / 'development_frozen.json').exists(): assert read_json(ROOT / 'development_frozen.json') == frozen
        else: write_json(ROOT / 'development_frozen.json', frozen)
        verify_files(files, ROOT); check_pause()
        for c in choices:
            check_pause(); row = c['record']; name = f"{row['branch']}_p{row['partition']}"
            jp = ROOT / 'final_test' / (name + '.json'); cp = jp.with_suffix('.csv')
            if jp.exists():
                old = read_json(jp)
                assert old['job'] == row['job'] and digest(cp) == old['prediction_sha256']
                continue
            status('final_evaluation', job=row['job'], completed_development=65)
            ck = torch.load(c['model_path'], map_location='cpu', weights_only=False)
            cfg = row['config']; rois = row['rois']; ids = plan['splits'][row['partition']]['indices']['test']
            assert ck['config'] == cfg and ck['epoch'] == 50 and ck['n_roi'] == len(rois)
            model = build_model('kan_legacy', len(rois), cfg).to('cuda'); geometry(model, len(rois), cfg)
            model.load_state_dict(ck['state'])
            prob = predict(model, input_features(x[ids][:, rois][:, :, rois], 'kan_legacy'), 'cuda', cfg['batch_size'])
            write_predictions(cp, ids, y[ids], prob)
            write_json(jp, dict(job=row['job'], variant=row['branch'], partition=row['partition'], candidate=row['candidate'],
                config=cfg, n_roi=len(rois), metrics=metrics(y[ids], prob), model_sha256=c['model_sha256'],
                prediction_sha256=digest(cp), development_frozen_sha256=digest(ROOT / 'development_frozen.json')))
            del model, ck; gc.collect(); torch.cuda.empty_cache()
        verify_files(plan['previous_artifacts'])
        status('evaluated', development_records=65, final_records=10,
               new_trainings=sum(not r['reused'] for r in rows))
    except SafePause:
        old = read_json(ROOT / 'status.json') if (ROOT / 'status.json').exists() else {}
        saved = None
        if (ROOT / 'state/current.pt').exists():
            ck = torch.load(ROOT / 'state/current.pt', map_location='cpu', weights_only=False)
            saved = dict(job=ck['job'], epoch=ck['epoch']); del ck
        status('paused', previous=old, saved_checkpoint=saved); sys.exit(75)
    except Exception:
        status('failed', error=traceback.format_exc()); raise
    finally: lock.close()

if __name__ == '__main__': main()
