import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ENSEMBLE_GATE_THRESHOLD,
    ENSEMBLE_WEIGHT_PATH,
    EQUIP_STATS_PATH,
    FINAL_EARLY_STOP_FRAC,
    LANE_STATS_PATH,
    MODEL_LGBM_PATH,
    MODEL_XGB_PATH,
    OPTUNA_TRAIN_END,
    OPTUNA_VAL_START,
    RANDOM_STATE,
    SANITY_START,
    TRAIN_PATH,
)
from features import make_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_splits(path):
    df = pd.read_csv(path, parse_dates=["date"])
    train_opt = df[df["date"] < OPTUNA_TRAIN_END].copy()
    sanity = df[(df["date"] >= SANITY_START) & (df["date"] < OPTUNA_VAL_START)].copy()
    val_opt = df[df["date"] >= OPTUNA_VAL_START].copy()
    full = df.copy()
    return train_opt, sanity, val_opt, full


def prepare(df_train, df_eval, lane_stats=None, equip_stats=None):
    X_train, ls, es = make_features(df_train, lane_stats, equip_stats)
    y_train = df_train["posted_rate"].values
    X_eval, _, _ = make_features(df_eval, ls, es)
    y_eval = df_eval["posted_rate"].values
    return X_train, y_train, X_eval, y_eval, ls, es


def tune_lgbm(X_train, y_train, X_val, y_val, n_trials=40):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "n_estimators": 1000,
            "subsample_freq": 1,
            "random_state": RANDOM_STATE,
            "verbosity": -1,
        }
        m = LGBMRegressor(**params)
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )
        preds = m.predict(X_val)
        return root_mean_squared_error(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def tune_xgb(X_train, y_train, X_val, y_val, n_trials=25):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "n_estimators": 1000,
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
            "verbosity": 0,
        }
        m = XGBRegressor(**params, early_stopping_rounds=50)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = m.predict(X_val)
        return root_mean_squared_error(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def train_final(df_full, lgbm_params, xgb_params, lane_stats_full, equip_stats_full):
    """Train final models on Jan-Oct using in-time 5% early-stopping slice."""
    df_full = df_full.sort_values("date")
    cutoff_idx = int(len(df_full) * (1 - FINAL_EARLY_STOP_FRAC))
    df_early_train = df_full.iloc[:cutoff_idx]
    df_early_stop = df_full.iloc[cutoff_idx:]

    X_et, y_et, X_es, y_es, _, _ = prepare(df_early_train, df_early_stop, lane_stats_full, equip_stats_full)

    # derive n_estimators from early-stop slice
    lgbm_es = LGBMRegressor(**{**lgbm_params, "n_estimators": 1000, "random_state": RANDOM_STATE, "verbosity": -1})
    lgbm_es.fit(X_et, y_et, eval_set=[(X_es, y_es)],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])
    n_lgbm = lgbm_es.best_iteration_

    xgb_es = XGBRegressor(**{**xgb_params, "n_estimators": 1000, "random_state": RANDOM_STATE,
                              "tree_method": "hist", "verbosity": 0, "early_stopping_rounds": 50})
    xgb_es.fit(X_et, y_et, eval_set=[(X_es, y_es)], verbose=False)
    n_xgb = xgb_es.best_iteration + 1  # XGBoost 2.x: best_iteration is 0-based index

    print(f"  Final n_estimators: LGBM={n_lgbm}, XGB={n_xgb}")

    # retrain on all Jan-Oct with fixed counts
    X_f, _, _ = make_features(df_full, lane_stats_full, equip_stats_full)
    y_f = df_full["posted_rate"].values

    lgbm_final = LGBMRegressor(**{**lgbm_params, "n_estimators": n_lgbm, "random_state": RANDOM_STATE, "verbosity": -1})
    lgbm_final.fit(X_f, y_f)

    xgb_final = XGBRegressor(**{**xgb_params, "n_estimators": n_xgb, "random_state": RANDOM_STATE,
                                 "tree_method": "hist", "verbosity": 0})
    xgb_final.fit(X_f, y_f)

    return lgbm_final, xgb_final


def main():
    print("Loading data...")
    train_opt, sanity, val_opt, full = load_splits(TRAIN_PATH)
    full_janoct = full[full["date"] < "2025-11-01"].copy()

    print("Preparing tuning features (Jan-Jul train, Sep-Oct val)...")
    X_tr, y_tr, X_val, y_val, ls_tune, es_tune = prepare(train_opt, val_opt)

    baseline_rmse = root_mean_squared_error(y_val, np.full_like(y_val, y_tr.mean()))
    print(f"Baseline RMSE (global mean): {baseline_rmse:.2f}")

    print("Tuning LightGBM (40 trials)...")
    lgbm_params, lgbm_val_rmse = tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=40)
    print(f"  Best LGBM val RMSE: {lgbm_val_rmse:.2f}")

    print("Tuning XGBoost (25 trials)...")
    xgb_params, xgb_val_rmse = tune_xgb(X_tr, y_tr, X_val, y_val, n_trials=25)
    print(f"  Best XGB  val RMSE: {xgb_val_rmse:.2f}")

    # Sanity check on August hold-out
    X_san, y_san, _, _, _, _ = prepare(sanity, sanity.iloc[:1], ls_tune, es_tune)
    lgbm_san = LGBMRegressor(**{**lgbm_params, "n_estimators": 500, "random_state": RANDOM_STATE, "verbosity": -1})
    lgbm_san.fit(X_tr, y_tr)
    san_rmse = root_mean_squared_error(y_san, lgbm_san.predict(X_san))
    print(f"Sanity (Aug) RMSE: {san_rmse:.2f}")
    if san_rmse > lgbm_val_rmse * 1.10:
        print("  WARNING: Aug RMSE >10% worse than val RMSE — possible Optuna bias to Sep-Oct window")

    # Ensemble weights
    lgbm_gain = baseline_rmse - lgbm_val_rmse
    xgb_gain = baseline_rmse - xgb_val_rmse
    w_lgbm = lgbm_gain / (lgbm_gain + xgb_gain)
    w_xgb = 1 - w_lgbm
    print(f"Ensemble weights: LGBM={w_lgbm:.3f}, XGB={w_xgb:.3f}")

    # Recompute lane stats on full Jan-Oct
    print("Recomputing lane stats on full Jan-Oct...")
    _, ls_full, es_full = make_features(full_janoct)

    print("Training final models on Jan-Oct...")
    lgbm_final, xgb_final = train_final(full_janoct, lgbm_params, xgb_params, ls_full, es_full)

    # Ensemble gate: evaluate on August sanity hold-out with final lane stats
    X_san2, y_san2, _, _, _, _ = prepare(sanity, sanity.iloc[:1], ls_full, es_full)
    lgbm_san2_rmse = root_mean_squared_error(y_san2, lgbm_final.predict(X_san2))
    ens_san_preds = w_lgbm * lgbm_final.predict(X_san2) + w_xgb * xgb_final.predict(X_san2)
    ens_san_rmse = root_mean_squared_error(y_san2, ens_san_preds)
    use_ensemble = (lgbm_san2_rmse - ens_san_rmse) / lgbm_san2_rmse > ENSEMBLE_GATE_THRESHOLD
    print(f"Aug RMSE — LGBM: {lgbm_san2_rmse:.2f}, Ensemble: {ens_san_rmse:.2f}")
    print(f"Using ensemble: {use_ensemble}")

    # Feature importance
    feat_names = lgbm_final.feature_name_
    importances = lgbm_final.feature_importances_
    top10 = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:10]
    print("\nTop 10 features (LGBM):")
    for name, imp in top10:
        print(f"  {name}: {imp}")

    # Serialize
    with open(MODEL_LGBM_PATH, "wb") as f:
        pickle.dump(lgbm_final, f)
    with open(MODEL_XGB_PATH, "wb") as f:
        pickle.dump(xgb_final, f)
    with open(LANE_STATS_PATH, "wb") as f:
        pickle.dump(ls_full, f)
    with open(EQUIP_STATS_PATH, "wb") as f:
        pickle.dump(es_full, f)
    with open(ENSEMBLE_WEIGHT_PATH, "wb") as f:
        pickle.dump({"w_lgbm": w_lgbm, "w_xgb": w_xgb, "use_ensemble": use_ensemble}, f)

    print("\nDone. Models saved.")


if __name__ == "__main__":
    main()
