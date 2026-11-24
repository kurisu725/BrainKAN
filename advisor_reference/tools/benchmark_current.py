"""Explicit, repeatable compute measurements; run when other GPU jobs have stopped.

Times cover synthetic GPU-resident tensors and model inference, or a complete
forward/backward/AdamW update. They exclude data I/O and permutation selection.
"""
from pathlib import Path
import sys,json,time,argparse,platform
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'advisor_code'))
from advisor_experiments import model_for,BASE
from brainkan.training import seed_everything

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-timing',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'evidence'/'compute_benchmark_rerun.json');a=p.parse_args()
    if a.run_timing and not torch.cuda.is_available():raise RuntimeError('CUDA is required for the recorded GPU timing protocol')
    torch.set_num_threads(4);results=[]
    for label,kind,n in [('Full KAN','kan',90),('KAN 64 ROI','kan',64),('KAN 30 ROI','kan',30),('MLP','mlp',90),('BrainNetCNN','brainnetcnn',90)]:
        print('Benchmark:',label,flush=True)
        seed_everything(0);model=model_for(kind,n,BASE);count=sum(p.numel() for p in model.parameters());row=dict(model=label,n_roi=n,parameters=count,batch_size=96)
        if a.run_timing:
            model.cuda();x=torch.randn(96,n*n,device='cuda');y=torch.arange(96,device='cuda')%2
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            for task in ('inference','train_step'):
                print('  ',task,flush=True)
                model.train(task=='train_step');times=[]
                def once():
                    if task=='inference':
                        with torch.no_grad():model(x)
                    else:
                        opt.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(model(x),y).backward();opt.step()
                for _ in range(10):once()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                for _ in range(50):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();once();end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end))
                row[task]=dict(median_ms=float(np.median(times)),p25_ms=float(np.percentile(times,25)),p75_ms=float(np.percentile(times,75)),
                    all_ms=times,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            del opt,x,y;model.cpu();torch.cuda.empty_cache()
        results.append(row)
    report=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),python=platform.python_version(),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,warmup=10,repeats=50,timing_requested=a.run_timing,
        scope='GPU resident synthetic Gaussian batch; excludes I/O, normalization and ROI estimation; no concurrent GPU experiment jobs',results=results)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps([{k:v for k,v in r.items() if k not in ('inference','train_step')} for r in results]))
if __name__=='__main__':main()
