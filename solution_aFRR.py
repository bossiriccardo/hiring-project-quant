import pandas as pd
import numpy as np
from tqdm import tqdm
from enum import StrEnum
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


"""
Script structure:
1) Model definition and training on historical data
2) Daily retraining backtest
3) Live prediction

Approach:
- Linear ridge regression
- One model per forecast horizon
- Features based on recent dynamics and intraday seasonality
"""


# ============================================================
# SETTINGS
# ============================================================
dataset_path="PowerSystemRightNow.csv"      # Historical minute level aFRR activation data by Energinet, used for model training and backtesting
zone = "DK2"
H = 8
LAM = 10.0
CLIP_Q = 0.995
TRAIN_DAYS = 56
TEST_DAYS  = 30

# ============================================================
# MODEL (data + features + targets + training)
# ============================================================

# Load dataset
df = pd.read_csv(dataset_path, sep=";", decimal=",")
cols = ["Minutes1DK", "aFRR_ActivatedDK1", "aFRR_ActivatedDK2"]
df = df[cols].copy()

df["Minutes1DK"] = pd.to_datetime(df["Minutes1DK"])
df = df.set_index("Minutes1DK").sort_index()

y = df[f"aFRR_Activated{zone}"].astype(float).dropna()

#Features definition
def features_def(y: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=y.index)
    out["y"] = y.astype(float)

    # Autoregressive features
    out["mean_15"] = out["y"].rolling(15, min_periods=15).mean()
    out["mean_45"] = out["y"].rolling(45, min_periods=45).mean()
    out["vol_15"]  = out["y"].rolling(15, min_periods=15).std()
    out["ewma_10"] = out["y"].ewm(span=10, adjust=False).mean()
    out["trend"]   = out["mean_15"] - out["mean_45"]

    # Seasonality (hour)
    hour = out.index.hour + out.index.minute / 60.0
    angle = 2 * np.pi * hour / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)

    return out.drop(columns=["y"])


# Build features df
X = features_def(y).dropna()

# convert into 15 min series
y_mtu = y.resample("15min").mean()


# timestamp for prediction time
base_delivery = (X.index + pd.Timedelta("1min")).ceil("15min")

#aligne the 15 minute series to the corresponding delivery periods for forecast horizon
targets = {}
for k in range(1, H + 1):
    delivery = base_delivery + pd.Timedelta(minutes=15 * (k - 1))
    targets[k] = y_mtu.reindex(delivery).to_numpy()


# Outlier handling
def clip_array(a: np.ndarray, q: float = CLIP_Q) -> np.ndarray:
    a2 = a.copy()
    m = ~np.isnan(a2)
    if m.sum() == 0:
        return a2
    lo, hi = np.quantile(a2[m], [1 - q, q])
    a2[m] = np.clip(a2[m], lo, hi)
    return a2

targets_clipped = {k: clip_array(v, q=CLIP_Q) for k, v in targets.items()}


# regression model (ridge)
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = LAM) -> np.ndarray:
    X1 = np.c_[np.ones(len(X)), X]          
    I = np.eye(X1.shape[1])
    I[0, 0] = 0.0                           
    beta = np.linalg.solve(X1.T @ X1 + lam * I, X1.T @ y)
    return beta

def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X1 = np.c_[np.ones(len(X)), X]
    return X1 @ beta


# models training 
X_np = X.values
betas = {}

for k in range(1, H + 1):
    yk = targets_clipped[k]
    mask = ~np.isnan(yk)
    betas[k] = ridge_fit(X_np[mask], yk[mask], lam=LAM)


# ============================================================
# check on last timestamp
# ============================================================
# x_last = X.values[-1:].copy()

# preds = {}
# for k in range(1, H + 1):
#     preds[k] = float(ridge_predict(x_last, betas[k])[0])

# now = X.index[-1]
# forecast_horizon = pd.date_range(start=(now + pd.Timedelta("1min")).ceil("15min"), periods=H, freq="15min")

# pred_series = pd.Series([preds[k] for k in range(1, H + 1)], index=forecast_horizon)
# print(pred_series.to_string())



# ============================================================
# Backtest
# ============================================================


# timestamp for prediction time and aligning the 15 minute series to the corresponding delivery periods for forecast horizon

def build_targets(y_mtu: pd.Series, minute_index: pd.DatetimeIndex, horizon: int = 8):
    base = (minute_index + pd.Timedelta("1min")).ceil("15min")
    out = {}
    for k in range(1, horizon + 1):
        delivery = base + pd.Timedelta(minutes=15 * (k - 1))
        out[k] = y_mtu.reindex(delivery).to_numpy()
    return out


def backtest_daily_retrain(df: pd.DataFrame):

    
    # Prepare  dataset (features + targets)

    y = df[f"aFRR_Activated{zone}"].astype(float).dropna()

    X_all = features_def(y).dropna()
    y_all = y.reindex(X_all.index)
    y_mtu_all = y.resample("15min").mean()

    # Define backtest window (last TEST_DAYS)
    
    last_time = X_all.index.max()

    first_test_day = last_time.floor("D") - pd.Timedelta(days=TEST_DAYS)
    min_possible   = (X_all.index.min() + pd.Timedelta(days=TRAIN_DAYS)).floor("D")

    first_test_day = max(first_test_day, min_possible)

    test_days = pd.date_range(start=first_test_day,end=last_time.floor("D"),freq="D")

    results = []

    
    # Daily retraining loop
    
    for day in tqdm(test_days, desc=f"Backtest {zone}"):

        train_start = day - pd.Timedelta(days=TRAIN_DAYS)
        train_end   = day
        test_start  = day
        test_end    = day + pd.Timedelta(days=1)

        
        # Train Set
        
        X_train = X_all.loc[train_start:train_end - pd.Timedelta("1ns")]
        if len(X_train) < 1000:
            continue

        y_train = y_all.loc[X_train.index]
        y_mtu_train = y_train.resample("15min").mean()

        # Build targets for training
        targets_train = build_targets(y_mtu_train, X_train.index, horizon=H)

        Xtr = X_train.values
        betas = {}

        for k in range(1, H + 1):
            yk = targets_train[k]
            mask = ~np.isnan(yk)

            if mask.sum() < 200:
                continue

            yk_clip = clip_array(yk[mask], q=CLIP_Q)
            betas[k] = ridge_fit(Xtr[mask], yk_clip, lam=LAM)

        
        # Test set (next day)
        
        X_test = X_all.loc[test_start:test_end - pd.Timedelta("1ns")]
        if len(X_test) == 0:
            continue

        targets_test = build_targets(y_mtu_all, X_test.index, horizon=H)
        Xte = X_test.values

        for k in range(1, H + 1):
            if k not in betas:
                continue

            pred = ridge_predict(Xte, betas[k])
            true = targets_test[k]

            valid = ~np.isnan(true)
            if valid.sum() == 0:
                continue

            results.append(pd.DataFrame({
                "day": day,
                "zone": zone,
                "horizon": k,
                "abs_error": np.abs(true[valid] - pred[valid]),
                "sign_correct": np.sign(true[valid]) == np.sign(pred[valid])
            }))

    if not results:
        return None, None, None

    
    # Aggregate metrics
    
    res = pd.concat(results, ignore_index=True)

    mae = res.groupby("horizon")["abs_error"].mean()
    sign_acc = res.groupby("horizon")["sign_correct"].mean()

    return res, mae, sign_acc


res1, mae1, sign1 = backtest_daily_retrain(df)
print("MAE\n", mae1)
print("SignAcc\n", sign1)





# ============================================================
# Model Applied to last RT values
# ============================================================


class BiddingZone(StrEnum):
    DK1 = "DK1"
    DK2 = "DK2"



def get_recent_afrr_activation_live(bidding_zone: str, minutes: int = 180) -> pd.Series:
    url = "https://api.energidataservice.dk/dataset/PowerSystemRightNow"

    now_utc = pd.Timestamp.utcnow()
    start_utc = (now_utc - pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M")

    r = requests.get(
        url,
        params={"start": start_utc}, verify=False, timeout=30
    )
    r.raise_for_status()
    js = r.json()

    s = (
        pd.Series(
            {
                pd.Timestamp(rec["Minutes1DK"]): rec[f"aFRR_Activated{bidding_zone}"]
                for rec in js["records"]
            }
        )
        .astype(float)
        .dropna()
        .sort_index()
    )
    return s



def predict_api(betas: dict[int, np.ndarray], zone: str = zone, minutes: int = 180) -> pd.Series:
    #minute level live series
    y_recent = get_recent_afrr_activation_live(zone, minutes=minutes)

    # Build features on the recent series (need 45 non NaN)
    X_recent = features_def(y_recent).dropna()
    if len(X_recent) == 0:
        raise ValueError(
            f"Not enough recent data to build features. "
            f"Try increasing minutes (current={minutes})."
        )

    # Use last available feature row
    x_last = X_recent.values[-1:]

    # Build forecast horizon
    now = X_recent.index[-1]
    forecast_horizon = pd.date_range(start=(now + pd.Timedelta("1min")).ceil("15min"),periods=H,freq="15min")

    # Predict each horizon using its own beta
    preds = [float(ridge_predict(x_last, betas[k])[0]) for k in range(1, H + 1)]

    return pd.Series(preds, index=forecast_horizon)

pred = predict_api(betas, zone=zone, minutes=180)
print(pred.to_string())
