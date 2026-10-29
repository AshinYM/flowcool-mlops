from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
PROFILE = os.getenv('PHM_PROFILE','single')
DATA = ROOT / ('data/phm2018_fleet' if PROFILE=='fleet' else 'data/phm2018')
STATE = Path(os.getenv('PHM_STATE_DIR', str(ROOT / ('phm_state_fleet' if PROFILE=='fleet' else 'phm_state')))).resolve()
STATE.mkdir(parents=True, exist_ok=True)
MODEL_NAME = 'WaferFlowcoolFleetLSTM' if PROFILE=='fleet' else 'WaferFlowcoolLSTM'
TOOLS = [f'{n:02d}M{m:02d}' for n in range(1,11) for m in [1,2]] if PROFILE=='fleet' else ['08M01']
TRACKING_URI = os.getenv('PHM_TRACKING_URI', f'sqlite:///{STATE / "mlflow.db"}')
FEATURES = ['FLOWCOOLPRESSURE', 'FLOWCOOLFLOWRATE', 'IONGAUGEPRESSURE',
            'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 'ETCHSUPPRESSORCURRENT',
            'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 'FLOWCOOLPRESSURE_MAX']
if PROFILE=='fleet':
    FEATURES=FEATURES[:-1]+['ETCHSUPPRESSORVOLTAGE','FIXTURETILTANGLE','ROTATIONSPEED',
        'ACTUALROTATIONANGLE','FIXTURESHUTTERPOSITION','ETCHSOURCEUSAGE',
        'ETCHAUXSOURCETIMER','ETCHAUX2SOURCETIMER','ACTUALSTEPDURATION']+['FLOWCOOLPRESSURE_MAX']
SEQ_LEN = 20
SEED = 42
