from pathlib import Path
import psutil
from storage import read_json

def dependency_state(root):
    root=Path(root)
    if (root/'PAUSE').exists():return False,'previous_experiment_paused'
    def read(name):return read_json(root/name) if (root/name).exists() else {}
    if read('status.json').get('stage')!='complete':return False,'previous_experiment_running_or_pending'
    if not read('validation.json').get('passed'):return False,'previous_validation_pending'
    if read('controller_status.json').get('stage')!='complete':return False,'previous_finalizer_pending'
    for p in psutil.process_iter(['name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or []).lower().replace('/','\\')
            if 'python' in (p.info['name'] or '').lower() and str(root).lower().replace('/','\\') in cmd:
                return False,'previous_process_exiting'
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return True,'ready'
