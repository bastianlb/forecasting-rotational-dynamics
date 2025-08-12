# config.py
from yacs.config import CfgNode as CN


def get_cfg_defaults():
    """Get default configuration"""
    _C = CN()
    _C.DATA_ROOT = "./data"

    # Simulation parameters
    _C.SIM = CN()
    _C.SIM.SEED = 42
    _C.SIM.N_SAMPLES = 8000
    _C.SIM.DT = 0.001
    _C.SIM.T_FINAL = 10.0
    _C.SIM.OMEGA_SCALE = 0.5
    _C.SIM.MIN_OMEGA_NORM = 0.1
    _C.SAVE_DERIVATIVES = True
    _C.SIM.OUTPUT_DIR = "./data/"

    # Rigid body parameters
    _C.BODY = CN()
    _C.BODY.MOI_DISTRIBUTIONS = [
        {"BASE": [0.1, 0.2, 0.3], "NOISE": 0.0},
        {"BASE": [0.1, 0.2, 0.3], "NOISE": 0.0},
        {"BASE": [0.1, 0.2, 0.3], "NOISE": 0.0},
        {"BASE": [0.1, 0.2, 0.3], "NOISE": 0.0},
    ]

    _C.BODY.VARIABLE_INERTIA = False
    _C.BODY.INERTIA_NOISE = 0.0

    # Scenario configurations
    _C.SCENARIOS = CN()

    # Free rotation scenario
    _C.SCENARIOS.FREE_ROTATION = CN()
    _C.SCENARIOS.FREE_ROTATION.ENABLED = False

    # Magnetic scenario
    _C.SCENARIOS.MAGNETIC = CN()
    _C.SCENARIOS.MAGNETIC.ENABLED = False
    _C.SCENARIOS.MAGNETIC.B_FIELD = [0.0, 0.0, 1.0]
    _C.SCENARIOS.MAGNETIC.MAGNETIC_MOMENT = None

    # Restoring scenario
    _C.SCENARIOS.RESTORING = CN()
    _C.SCENARIOS.RESTORING.ENABLED = False
    _C.SCENARIOS.RESTORING.K = 0.1
    _C.SCENARIOS.RESTORING.REFERENCE_QUAT = None

    # Damped scenario
    _C.SCENARIOS.DAMPED = CN()
    _C.SCENARIOS.DAMPED.ENABLED = False
    _C.SCENARIOS.DAMPED.DAMPING_MATRIX = [[-0.1, 0.0, 0.0],
                                         [0.0, -0.1, 0.0],
                                         [0.0, 0.0, -0.1]]
    _C.SCENARIOS.DAMPED.NATURAL_FREQUENCY = None

    # Combined damped magnetic scenario
    _C.SCENARIOS.DAMPED_MAGNETIC = CN()
    _C.SCENARIOS.DAMPED_MAGNETIC.ENABLED = False
    _C.SCENARIOS.DAMPED_MAGNETIC.WEIGHTS = [1.0, 1.0]

    # Constant acceleration scenario
    _C.SCENARIOS.CONSTANT_ACC = CN()
    _C.SCENARIOS.CONSTANT_ACC.ENABLED = False
    _C.SCENARIOS.CONSTANT_ACC.ACCELERATION = [0.1, 0.0, 0.0]
    
    # Linear acceleration scenario
    _C.SCENARIOS.LINEAR_ACC = CN()
    _C.SCENARIOS.LINEAR_ACC.ENABLED = False
    _C.SCENARIOS.LINEAR_ACC.A_MATRIX = [[-0.1, 0.0, 0.0],
                                       [0.0, -0.1, 0.0],
                                       [0.0, 0.0, -0.1]]
    _C.SCENARIOS.LINEAR_ACC.BIAS = None

    return _C.clone()
