from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).parent.parent
TRAIN_PATH = ROOT / "train-test.csv"
VAL_PATH = ROOT / "validation.csv"
DEC_PATH = ROOT / "december-chart-inputs.csv"
VAL_PRED_TEMPLATE = ROOT / "validation-predictions-template.csv"
VAL_PRED_OUT = ROOT / "validation_predictions.csv"
MODEL_LGBM_PATH = ROOT / "freight" / "model_lgbm.pkl"
MODEL_XGB_PATH = ROOT / "freight" / "model_xgb.pkl"
LANE_STATS_PATH = ROOT / "freight" / "lane_stats.pkl"
EQUIP_STATS_PATH = ROOT / "freight" / "equip_stats.pkl"
ENSEMBLE_WEIGHT_PATH = ROOT / "freight" / "ensemble_weight.pkl"

# Date cutoffs
OPTUNA_TRAIN_END = "2025-08-01"    # Jan–Jul exclusive upper bound
SANITY_START = "2025-08-01"        # Aug start
OPTUNA_VAL_START = "2025-09-01"    # Sep–Oct start

FINAL_EARLY_STOP_FRAC = 0.05       # last 5% of Jan-Oct for final n_estimators
ENSEMBLE_GATE_THRESHOLD = 0.005    # min relative RMSE improvement on Aug to use ensemble
LANE_SMOOTH_K = 5                  # Bayesian smoothing k
