from pathlib import Path
import sys,json,copy,datetime,shutil,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent;sys.path.insert(0,str(ROOT/'code'))
from storage import write_json,read_json,digest
def main():
    assert not (ROOT/'plan.json').exists(),'Plan already frozen'
    splits=read_json(B/'Random_split_comparison_20261027/split_manifest.json')['splits']
    splits=copy.deepcopy(splits)
    data=np.load(ROOT/'data/abide_dparsf.npz')
    for s in splits:
        s['indices']['development']=s['indices'].pop('valid');idx=s['indices'];tr=idx['train']
        assert [len(idx[k]) for k in ['train','development','test']]==[709,89,84]
        assert len(set(sum(idx.values(),[])))==882
        assert set(data['sites'][idx['development']])|set(data['sites'][idx['test']])<=set(data['sites'][tr])
    specs={
      'fbnetgen':[dict(embedding_size=16,batch_size=16,lr=1e-4,weight_decay=1e-4,epochs=500)],
      'brainnetgnn':[dict(batch_size=32,lr=5e-4,weight_decay=0.,epochs=3000,early_stop_patience=15,scheduler_patience=10,l1=1e-5)],
      'res_dae':[dict(n_input=2002,hidden=h,latent=l,dropout=.5,batch_size=64,lr=lr,weight_decay=0.,epochs=250,joint_epochs=200)
                 for h,l in [(256,64),(512,128)] for lr in [1e-4,1e-3]]}
    jobs=[dict(id=f'partition_{s["id"]}_{kind}_candidate_{i}',partition=s['id'],kind=kind,config=cfg)
           for s in splits for kind,configs in specs.items() for i,cfg in enumerate(configs)]
    plan=dict(created=datetime.datetime.now().astimezone().isoformat(),dependency=str(B/'KAN_crossfit_selection_20261108'),
      dataset=dict(subjects=882,atlas='AAL first 90',source='DPARSF/filt_global/rois_aal',timepoints=[77,315],data_sha256=digest(ROOT/'data/abide_dparsf.npz'),
                   mat_sha256=digest(B/'BrainKAN-complete/data/ALLASD1_aal.mat'),role='same subjects and exact FC-matched time series, no new cohort'),
      splits=splits,training_seed=1,jobs=jobs,expected_candidate_fits=30,expected_final_records=15,
      rules=dict(fbnetgen='Official GRU/product graph/group loss/mixup/sparsity; embedding16 from manuscript; final500; train-only scalar normalization; packed valid 4-frame windows',
        brainnetgnn='Official CNN/attention/GatedGraphConv/TopK/Set2Set; AAL90, adaptive temporal pooling; all forward branches in optimizer; original L1; paper patience10 scheduler and15 early stop; best development objective',
        res_dae='Paper-based residual denoising autoencoder; train-only ComBat(age/sex), Fisher top50%, EROS-SMOTE; 4 development-selected candidates, joint200/classifier50; development Youden threshold grid0:0.1:1',
        res_unspecified_parameters='Widths, LR, epochs, dropout, k=5 and balancing ratio are explicit reimplementation settings; not recovered author hyperparameters. Threshold uses outer development instead of paper training CV.',
        candidate_selection='Res-DAE development AUC descending, binary log loss ascending, candidate identifier ascending',
        test_gate='All30 candidate fits across all5 partitions and all15 choices sealed before test prediction',
        storage='One rolling restart checkpoint; one selected model per model/partition; superseded candidate weights not retained',
        presentation='Seven original baselines plus one BrainKAN EN rank Top75; prior display choice stays fixed',
        scope='Five repeated stratified random holdouts, not five-fold CV; previously inspected test cohorts, exploratory comparison'),
      sources=read_json(ROOT/'sources/provenance.json'))
    shutil.copy2(B/'Results_tables_20261108/tables.json',ROOT/'reference_tables.json')
    old={str(B/'Results_tables_20261108/tables.json'):digest(B/'Results_tables_20261108/tables.json'),str(B/'BrainKAN-complete/data/ALLASD1_aal.mat'):plan['dataset']['mat_sha256']}
    for folder in ['Baseline_rerun_20261106','Random_split_comparison_20261027','ROI_selection_study_20261102']:
        for name in ['plan.json','comparison.json','split_manifest.json']:
            p=B/folder/name
            if p.exists():old[str(p)]=digest(p)
        for p in (B/folder/'final_test').glob('*'):
            if p.is_file():old[str(p)]=digest(p)
    # Preserve the active experiment's immutable setup only; its results are still being written.
    for p in [B/'KAN_crossfit_selection_20261108/plan.json',*(B/'KAN_crossfit_selection_20261108/code').rglob('*.py')]:old[str(p)]=digest(p)
    write_json(ROOT/'previous_artifacts.json',old);write_json(ROOT/'plan.json',plan)
    paths=[ROOT/'plan.json',ROOT/'data/abide_dparsf.npz',ROOT/'reference_tables.json',ROOT/'previous_artifacts.json',ROOT/'audit/input_validation.json',ROOT/'sources/provenance.json',ROOT/'control.py',ROOT/'run_background.py']
    paths+=list((ROOT/'code').rglob('*.py'))
    write_json(ROOT/'frozen_manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)})
    write_json(ROOT/'status.json',dict(stage='queued',time=plan['created'],dependency=plan['dependency'],expected_candidate_fits=30,expected_final_records=15))
    import torch,scipy,sklearn,pandas,torch_geometric
    write_json(ROOT/'audit/environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pandas.__version__,torch_geometric=torch_geometric.__version__))
    print('Frozen: 30 candidate fits, 15 final evaluations, 5 original partitions. No training started.')
if __name__=='__main__':main()
