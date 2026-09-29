import numpy as np
import pandas as pd
from config import LANE_SMOOTH_K


def make_features(df: pd.DataFrame, lane_stats: dict | None = None, equip_stats: dict | None = None):
    """
    Build feature matrix. Pass None for lane_stats/equip_stats at training time
    (they'll be computed from df and returned). Pass dicts at inference time.
    Returns (feature_df, lane_stats, equip_stats).
    quote_signal*distance is never included.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # --- temporal ---
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    # ISO week 1 can appear in late December (year-boundary artifact); treat as week 52
    df.loc[(df["date"].dt.month == 12) & (df["woy"] == 1), "woy"] = 52
    df["quarter"] = df["date"].dt.quarter
    df["is_month_start"] = (df["dom"] <= 3).astype(int)
    df["is_month_end"] = (df["dom"] >= df["date"].dt.days_in_month - 2).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # --- geographic ---
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["geo_mid_lat"] = (df["pickup_lat"] + df["delivery_lat"]) / 2
    df["geo_mid_lon"] = (df["pickup_lon"] + df["delivery_lon"]) / 2

    # --- equipment one-hot ---
    for eq in ["Dry Van", "Reefer", "Flatbed"]:
        col = "equip_" + eq.lower().replace(" ", "_")
        df[col] = (df["equipment"] == eq).astype(int)

    # --- log distance (guard against <=0) ---
    df["log_distance"] = np.log(df["distance"].clip(lower=1.0))

    # --- market_index imputation (training fold monthly median, applied inline) ---
    if df["market_index"].isna().any():
        monthly_med = df.groupby("month")["market_index"].transform("median")
        df["market_index"] = df["market_index"].fillna(monthly_med)
        # fallback for any still-missing
        df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    # --- interactions ---
    df["dist_x_market"] = df["distance"] * df["market_index"]
    df["dist_x_reefer"] = df["distance"] * df["equip_reefer"]

    # --- lane target encoding ---
    df["lane"] = list(zip(df["pickup"], df["delivery"]))
    df["rpm"] = df["posted_rate"] / df["distance"].clip(lower=1.0) if "posted_rate" in df.columns else np.nan

    if equip_stats is None:
        equip_stats = df.groupby("equipment")["rpm"].mean().to_dict()

    if lane_stats is None:
        lane_stats = _compute_lane_stats(df, equip_stats)

    global_mean = np.mean(list(equip_stats.values())) if equip_stats else 0.0
    df["lane_enc"] = df["lane"].map(lane_stats).fillna(df["equipment"].map(equip_stats)).fillna(global_mean)

    feature_cols = [
        "distance", "weight", "market_index", "quote_signal",
        "month", "dow", "dom", "woy", "quarter",
        "is_month_start", "is_month_end",
        "month_sin", "month_cos", "dow_sin", "dow_cos",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "lat_diff", "lon_diff", "geo_mid_lat", "geo_mid_lon",
        "equip_dry_van", "equip_reefer", "equip_flatbed",
        "log_distance", "dist_x_market", "dist_x_reefer",
        "lane_enc",
    ]
    return df[feature_cols], lane_stats, equip_stats


def _compute_lane_stats(df: pd.DataFrame, equip_stats: dict) -> dict:
    assert "rpm" in df.columns, "_compute_lane_stats requires posted_rate column"
    stats = {}
    lane_groups = df.groupby("lane")
    for lane, grp in lane_groups:
        n = len(grp)
        lane_mean = grp["rpm"].mean()
        eq = grp["equipment"].mode()[0]
        eq_mean = equip_stats.get(eq, df["rpm"].mean())
        stats[lane] = (n * lane_mean + LANE_SMOOTH_K * eq_mean) / (n + LANE_SMOOTH_K)
    return stats


