import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from scipy.sparse import hstack

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models/db_anomaly_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/query_vectorizer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/feature_scaler.pkl"))

weekday_freq = joblib.load(os.path.join(BASE_DIR, "models/weekday_freq.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "models/weekday_threshold.pkl"))
freq_maps = joblib.load(os.path.join(BASE_DIR, "models/freq_maps.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "models/feature_columns.pkl"))
feature_median = joblib.load(os.path.join(BASE_DIR, "models/feature_median.pkl"))
feature_std = joblib.load(os.path.join(BASE_DIR, "models/feature_std.pkl"))
query_freq_map = joblib.load(os.path.join(BASE_DIR, "models/query_freq.pkl"))

def predict_new_data(input_df: pd.DataFrame):
    df = input_df.copy()

    # -----------------------------
    # Preprocessing
    # -----------------------------
    df["DB_username"] = df["DB_username"].str.lower()
    df["OS_username"] = df["OS_username"].str.lower()
    df["Source_programname"] = df["Source_programname"].str.lower()

    df["audit_at"] = pd.to_datetime(df["audit_at"], format="%d-%m-%Y %H:%M")
    df["hour"] = df["audit_at"].dt.hour
    df["day_of_week"] = df["audit_at"].dt.dayofweek
    df = df.drop(columns=["audit_at"])

    # -----------------------------
    # Load saved stats
    # -----------------------------
    # weekday_freq = joblib.load("weekday_freq.pkl")
    # threshold = joblib.load("weekday_threshold.pkl")
    # freq_maps = joblib.load("freq_maps.pkl")
    # feature_cols = joblib.load("feature_columns.pkl")
    # feature_median = joblib.load("feature_median.pkl")
    # feature_std = joblib.load("feature_std.pkl")

    # -----------------------------
    # Day anomaly
    # -----------------------------
    df["weekday_frequency"] = df["day_of_week"].map(weekday_freq).fillna(0)
    df["is_day_anomaly"] = (df["weekday_frequency"] < threshold).astype(int)

    # query_freq_map = joblib.load("query_freq.pkl")

    df["query_frequency"] = df["full_qry_hv"].map(query_freq_map).fillna(0)

    # -----------------------------
    # Frequency encoding
    # -----------------------------
    for col in ["Src_ip", "Source_programname", "OS_username", "DB_username"]:
        df[col] = df[col].map(freq_maps[col]).fillna(0)

    # -----------------------------
    # Load model artifacts
    # -----------------------------
    # model = joblib.load("db_anomaly_model.pkl")
    # vectorizer = joblib.load("query_vectorizer.pkl")
    # scaler = joblib.load("feature_scaler.pkl")

    # -----------------------------
    # Transform
    # -----------------------------
    query_features = vectorizer.transform(df["full_query"])

    for col in feature_cols:
        if col not in df:
            df[col] = 0

    numeric_features = df[feature_cols]
    numeric_matrix = scaler.transform(numeric_features)

    X = hstack([numeric_matrix, query_features])
    X_dense = X.toarray()

    # -----------------------------
    # Predict
    # -----------------------------
    labels = model.predict(X_dense)
    lof_scores = -model.decision_function(X_dense)

    # -----------------------------
    # Reason logic
    # -----------------------------
    feature_cols_reason = [
        "query_frequency",
        "Src_ip",
        "Source_programname",
        "OS_username",
        "hour",
        "DB_username"
    ]

    reason_labels = {
        "query_frequency":    "Unusual SQL query pattern",
        "Src_ip":             "Unusual source IP behaviour",
        "Source_programname": "Unusual program access pattern",
        "OS_username":        "Unusual OS user behaviour",
        "hour":               "Unusual access time",
        "DB_username":        "Unusual DB username",
    }

    reasons = []

    # ML-based reason
    if labels[0] == -1:
        for col in feature_cols_reason:
            if feature_std[col] == 0:
                continue

            val = df[col].iloc[0]
            median = feature_median[col]
            std = feature_std[col]

            deviation = abs(val - median) / std

            if deviation > 2:
                reasons.append(reason_labels[col])

        if not reasons:
            reasons.append(f"Model detected unusual behaviour (LOF score: {lof_scores[0]:.2f})")

    # Rule-based reason
    if df["is_day_anomaly"].iloc[0] == 1:
        reasons.append("Access on rare / low-activity day")

    if not reasons:
        reasons = ["Normal behaviour"]

    # -----------------------------
    # Final output
    # -----------------------------
    return {
        "anomaly_flag": int(labels[0]),
        "lof_score": float(lof_scores[0]),
        "reason": ", ".join(reasons)
    }