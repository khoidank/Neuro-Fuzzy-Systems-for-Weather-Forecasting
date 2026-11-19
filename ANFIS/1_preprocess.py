"""Tien xu ly du lieu Daily Climate (Delhi) cho bai toan du bao nhiet do ngay ke tiep.
Chay tu thu muc goc repo:  python 1_preprocess.py
Dau vao : data/raw/DailyDelhiClimateTrain.csv, DailyDelhiClimateTest.csv
Dau ra  : data/processed/X_train.npy, y_train.npy, X_test.npy, y_test.npy, scaler.pkl
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

RAW, OUT = "data/raw/", "data/processed/"
FEATS = ["meantemp", "humidity", "wind_speed", "meanpressure"]
os.makedirs(OUT, exist_ok=True)


def load(name):
    df = pd.read_csv(RAW + name, parse_dates=["date"]).sort_values("date")
    df = df.drop_duplicates("date").set_index("date")
    # meanpressure co gia tri ngoai lai bat thuong -> coi la NaN roi noi suy
    bad = (df["meanpressure"] < 980) | (df["meanpressure"] > 1040)
    print(f"{name}: {bad.sum()} gia tri meanpressure bat thuong")
    df.loc[bad, "meanpressure"] = np.nan
    df[FEATS] = df[FEATS].interpolate().bfill().ffill()
    df["target"] = df["meantemp"].shift(-1)  # nhiet do ngay ke tiep
    return df.dropna(subset=["target"])


def main():
    train = load("DailyDelhiClimateTrain.csv")
    test = load("DailyDelhiClimateTest.csv")

    scaler = MinMaxScaler().fit(train[FEATS])  # chi fit tren train
    tmin, tmax = scaler.data_min_[0], scaler.data_max_[0]
    scale_y = lambda y: (y - tmin) / (tmax - tmin)  # cung thang voi meantemp

    arrays = {
        "X_train": scaler.transform(train[FEATS]),
        "y_train": scale_y(train["target"].values),
        "X_test": scaler.transform(test[FEATS]),
        "y_test": scale_y(test["target"].values),
    }
    for k, v in arrays.items():
        assert not np.isnan(v).any(), f"{k} con NaN"
        np.save(f"{OUT}{k}.npy", v)
        print(k, v.shape, f"min={v.min():.3f} max={v.max():.3f}")
    joblib.dump(scaler, OUT + "scaler.pkl")
    print("Xong. Khoi phuc don vi that: y * (max - min) + min voi", tmin, tmax)


if __name__ == "__main__":
    main()
