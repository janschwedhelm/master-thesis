from pathlib import Path

# Try to locate the GP training/opt scripts
try:
    GP_TRAIN_FILE = str(Path(__path__[0]) / "gp_train.py")
    GP_OPT_FILE = str(Path(__path__[0]) / "gp_opt.py")
    GP_OPT_SAMPLING_FILE = str(Path(__path__[0]) / "gp_opt_sampling.py")
    DNGO_TRAIN_FILE = str(Path(__path__[0]) / "dngo/dngo_train.py")
    ENTMOOT_TRAIN_OPT_FILE = str(Path(__path__[0]) / "entmoot_bo_optimization.py")
except:
    pass
