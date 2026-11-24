from pathlib import Path
import psutil
from storage import read_json

def dependency_state(root):
    root=Path(root)
    if (root/'PAUSE').exists():return False,'FC baseline pause requested'
    if not (root/'status.json').exists():return False,'FC baseline status unavailable'
    if read_json(root/'status.json').get('state')!='complete':return False,'FC baseline training or finalization pending'
    if not (root/'validation.json').exists() or not read_json(root/'validation.json').get('passed'):return False,'FC validation pending'
    for p in psutil.process_iter(['name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or []).lower().replace('/','\\')
            if 'python' in (p.info['name'] or '').lower() and str(root).lower().replace('/','\\') in cmd and 'run_background.py' in cmd:
                return False,'FC worker exiting'
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return True,'ready'
