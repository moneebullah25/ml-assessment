import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEC_PATH,
    ENSEMBLE_WEIGHT_PATH,
    EQUIP_STATS_PATH,
    LANE_STATS_PATH,
    MODEL_LGBM_PATH,
    MODEL_XGB_PATH,
    TRAIN_PATH,
    VAL_PATH,
    VAL_PRED_OUT,
    VAL_PRED_TEMPLATE,
)
from features import make_features


def load_artifacts():
    with open(MODEL_LGBM_PATH, "rb") as f:
        lgbm = pickle.load(f)
    with open(MODEL_XGB_PATH, "rb") as f:
        xgb = pickle.load(f)
    with open(LANE_STATS_PATH, "rb") as f:
        lane_stats = pickle.load(f)
    with open(EQUIP_STATS_PATH, "rb") as f:
        equip_stats = pickle.load(f)
    with open(ENSEMBLE_WEIGHT_PATH, "rb") as f:
        ew = pickle.load(f)
    return lgbm, xgb, lane_stats, equip_stats, ew


def predict(df, lgbm, xgb, lane_stats, equip_stats, ew):
    X, _, _ = make_features(df, lane_stats, equip_stats)
    if ew["use_ensemble"]:
        preds = ew["w_lgbm"] * lgbm.predict(X) + ew["w_xgb"] * xgb.predict(X)
    else:
        preds = lgbm.predict(X)
    return np.clip(preds, a_min=1.0, a_max=None)


def impute_december_signals(dec_df, train_path):
    """
    december-chart-inputs.csv has no market_index/quote_signal/coordinates.
    Impute using October means for this specific lane+equipment from training data.
    """
    train = pd.read_csv(train_path, parse_dates=["date"])
    pickup = dec_df["pickup"].iloc[0]
    delivery = dec_df["delivery"].iloc[0]
    equipment = dec_df["equipment"].iloc[0]
    lane_mask = (
        (train["pickup"] == pickup)
        & (train["delivery"] == delivery)
        & (train["equipment"] == equipment)
        & (train["date"].dt.month == 10)
    )
    oct_lane = train[lane_mask]
    # fall back to all October if lane has no October rows
    oct_all = train[train["date"].dt.month == 10]
    market_index = oct_lane["market_index"].mean() if len(oct_lane) else oct_all["market_index"].mean()
    quote_signal = oct_lane["quote_signal"].mean() if len(oct_lane) else oct_all["quote_signal"].mean()

    # Use coordinates from training data for this lane (not city-center approximations)
    lane_rows = train[(train["pickup"] == pickup) & (train["delivery"] == delivery)]
    pickup_lat = lane_rows["pickup_lat"].mean()
    pickup_lon = lane_rows["pickup_lon"].mean()
    delivery_lat = lane_rows["delivery_lat"].mean()
    delivery_lon = lane_rows["delivery_lon"].mean()

    dec_df = dec_df.copy()
    dec_df["market_index"] = market_index
    dec_df["quote_signal"] = quote_signal
    dec_df["pickup_lat"] = pickup_lat
    dec_df["pickup_lon"] = pickup_lon
    dec_df["delivery_lat"] = delivery_lat
    dec_df["delivery_lon"] = delivery_lon
    return dec_df


def main():

    lgbm, xgb, lane_stats, equip_stats, ew = load_artifacts()

    # --- validation predictions ---
    print("Predicting validation.csv...")
    val = pd.read_csv(VAL_PATH, parse_dates=["date"])
    val_preds = predict(val, lgbm, xgb, lane_stats, equip_stats, ew)

    template = pd.read_csv(VAL_PRED_TEMPLATE)
    assert list(val["load_id"]) == list(template["load_id"]), "load_id order mismatch between val and template"
    template["predicted_rate"] = val_preds
    template.to_csv(VAL_PRED_OUT, index=False)
    print(f"Saved {len(template)} predictions to {VAL_PRED_OUT}")

    # --- december predictions ---
    print("Predicting december-chart-inputs.csv...")
    dec = pd.read_csv(DEC_PATH, parse_dates=["date"])
    dec = impute_december_signals(dec, TRAIN_PATH)
    dec_preds = predict(dec, lgbm, xgb, lane_stats, equip_stats, ew)

    # score.py expects exactly these 7 cols; re-read original to avoid carrying imputed cols
    dec_orig = pd.read_csv(DEC_PATH)
    dec_orig["predicted_rate"] = dec_preds
    dec_orig[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].to_csv(DEC_PATH, index=False)
    print(f"Wrote December predictions to {DEC_PATH}")

    # --- run scorer ---
    print("Running score.py...")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "score.py"),
         "--predictions", str(VAL_PRED_OUT),
         "--december-predictions", str(DEC_PATH)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("SCORER ERROR:", result.stderr)
        sys.exit(result.returncode)
    else:
        print("Scorer passed.")


if __name__ == "__main__":
    main()
