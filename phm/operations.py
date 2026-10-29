"""Read actual artifacts and events for the operator dashboard."""
import json
import time
import sqlite3
from datetime import datetime, timezone
from collections import deque
from phm.config import STATE, PROFILE, TOOLS, MODEL_NAME
from phm import runtime


def read_events(limit=100):
    path = STATE / 'aiops.jsonl'
    if not path.exists():
        return []
    with runtime.LOCK:
        with path.open() as source:
            return [json.loads(line) for line in deque(source, maxlen=limit)]


def summary(jobs):
    reports = []
    registry_times={}
    database=STATE/'mlflow.db'
    if database.exists():
        from urllib.parse import quote
        with sqlite3.connect(f'file:{quote(str(database))}?mode=ro',uri=True) as connection:
            registry_times={str(v):datetime.fromtimestamp(t/1000,timezone.utc).isoformat()
                            for v,t in connection.execute('SELECT version,creation_time FROM model_versions WHERE name=?',(MODEL_NAME,)) if t}
    for path in (STATE / 'bundles').glob('*/report.json'):
        report = json.loads(path.read_text())
        reports.append({k: report.get(k) for k in
                        ['version', 'run_id', 'mode', 'validation', 'gate_passed',
                         'epochs_completed', 'train_sequences', 'seconds', 'registered_at', 'training_focus']})
        reports[-1]['registered_at']=report.get('registered_at') or registry_times.get(str(report['version']))
    reports.sort(key=lambda r: int(r['version']), reverse=True)
    try:
        champion = runtime.pointer()
    except FileNotFoundError:
        champion = None
    if champion:
        champion['registered_at']=champion.get('registered_at') or registry_times.get(str(champion['version']))
    active_version = champion['version'] if champion else None
    for report in reports:
        report['deployment'] = ('active' if report['version'] == active_version else
                                'validated_history' if report['gate_passed'] else 'rejected')
    events = read_events(300)
    trigger = next((e for e in reversed(events) if e['kind'] == 'retrain_triggered'), None)
    outcome = None
    job = None
    if trigger:
        job = jobs.get(trigger['job_id'])
        outcome = next((e for e in reversed(events)
                        if e.get('job_id') == trigger['job_id'] and e['kind'] in
                        ['candidate_promoted', 'candidate_rejected', 'retrain_failed']), None)
    status = job['status'] if job else ('completed' if outcome and outcome['kind'] != 'retrain_failed'
                                       else 'failed' if outcome else 'running' if trigger else 'not_run')
    success = outcome and outcome['kind'] == 'candidate_promoted'
    rejected = outcome and outcome['kind'] == 'candidate_rejected'
    alert = next((e for e in reversed(events) if e['kind'] == 'monitor_alert' and
                  (not trigger or e['at'] <= trigger['at'])), None)
    upload = next((e for e in reversed(events) if e['kind'] in ['csv_uploaded','dataset_connected','dataset_selected'] and
                   (e.get('upload_id') == trigger.get('source_upload_id') if
                    trigger and trigger.get('source_upload_id') else
                    not trigger or e['at'] <= trigger['at'])), None)
    stages = [
        {'name': '데이터 수집', 'status': 'completed' if upload else 'not_run',
         'detail': f"연결 데이터 {upload['rows']:,}행" if upload else '데이터 연결 대기'},
        {'name': '입력·품질 감시', 'status': 'completed' if alert else 'not_run',
         'detail': '실제 관측 기록' if alert else '관측 배치 대기'},
        {'name': '변화 감지', 'status': 'alert' if alert else 'not_run',
         'detail': alert['status'] if alert else '감지 기록 없음'},
        {'name': '재학습 트리거', 'status': 'completed' if trigger else 'not_run',
         'detail': trigger.get('mode', '') if trigger else '대응 대기'},
        {'name': '모델 학습', 'status': status,
         'detail': '확정 train · 기존 가중치' if trigger else '학습 대기'},
        {'name': '검증 게이트', 'status': 'completed' if success else 'rejected' if rejected else
                                         'failed' if status == 'failed' else 'not_run',
         'detail': '통과' if success else '후보 차단' if rejected else '고정 validation'},
        {'name': '서빙 반영', 'status': 'completed' if success else 'retained' if rejected else 'not_run',
         'detail': f'현재 v{active_version}' if active_version else '검증 모델 없음'},
    ]
    with runtime.LOCK:
        quality = list(runtime.QUALITY)
        recent = list(runtime.REQUESTS['recent'])
    evaluation_path = STATE / 'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
    return {'profile': PROFILE, 'supported_tools': TOOLS,
            'request_bins': [sum(0 <= time.time()-r['ts']-offset < 60 for r in recent) for offset in [240,180,120,60,0]],
            'champion': {k: v for k, v in champion.items() if k != 'history'} if champion else None,
            'model_history': reports, 'service_metrics': runtime.service_metrics(),
            'latency_samples': [{'ts': r['ts'], 'ms': r['duration_ms'], 'status': r['status']} for r in recent[-24:]],
            'confirmed_label_count': len(quality), 'pipeline': stages,
            'pipeline_job_id': trigger['job_id'] if trigger else None,
            'pipeline_status': status, 'events': events[-20:],
            'evaluation': evaluation,
            'scope': 'Historical equipment replay; phases come from recorded events, not a scheduler.'}
