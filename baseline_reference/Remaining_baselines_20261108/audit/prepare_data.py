from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
from scipy.io import loadmat
R=Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(R/'code'))
from storage import digest,write_json,save_npz
def main():
    root=B/'Advisor_revision_20261026/audit/dparsf_filt_global'
    verification=json.loads((root/'verification.json').read_text())
    records={v['subject_id']:v for v in verification['records']}
    matpath=B/'BrainKAN-complete/data/ALLASD1_aal.mat';mat=loadmat(matpath)
    fc=np.concatenate([mat['net_'+k] for k in ['train','valid','test']])
    ph=np.concatenate([mat['phenotype_'+k] for k in ['train','valid','test']])
    ids=ph[:,0].astype(np.int64);labels=ph[:,2].astype(np.int64)
    assert len(ids)==882 and len(set(ids))==882 and set(labels)=={0,1}
    ts=np.zeros((882,90,315),np.float32);lengths=[];manifest=[]
    for i,sid in enumerate(ids):
        rec=records[int(sid)];path=root/rec['url'].split('/')[-1]
        assert digest(path)==rec['sha256']
        raw=np.loadtxt(path)[:,:90];err=float(abs(np.corrcoef(raw,rowvar=False)-fc[i]).max());assert err<1e-10,(sid,err)
        n=len(raw);ts[i,:,:n]=raw.T.astype(np.float32);lengths.append(n)
        manifest.append(dict(index=i,subject_id=int(sid),file=str(path),sha256=rec['sha256'],timepoints=n,fc_max_abs_difference=err))
    save_npz(R/'data/abide_dparsf.npz',time_series=ts,lengths=np.asarray(lengths),fc=fc.astype(np.float32),
             labels=labels,subject_ids=ids,sites=ph[:,1].astype(np.int64),phenotypes=ph)
    write_json(R/'audit/input_validation.json',dict(passed=True,n=882,n_roi=90,pipeline='DPARSF/filt_global/rois_aal; first 90 ROIs',
        length_min=min(lengths),length_max=max(lengths),exact_fc_matches=882,max_abs_difference=max(v['fc_max_abs_difference'] for v in manifest),
        mat_path=str(matpath),mat_sha256=digest(matpath),data_sha256=digest(R/'data/abide_dparsf.npz'),records=manifest))
    print(dict(n=882,exact_fc_matches=882,min_frames=min(lengths),max_frames=max(lengths),max_abs_difference=max(v['fc_max_abs_difference'] for v in manifest)))
if __name__=='__main__':main()
