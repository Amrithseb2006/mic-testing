import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from scipy.sparse import hstack
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DAY_ANOMALY_BOOST = 25          # Risk score boost for rule-based day anomaly
DAY_ANOMALY_QUANTILE = 0.10     # Bottom 10% weekday frequency = anomalous day
LOF_N_NEIGHBORS = 20
LOF_CONTAMINATION = 0.02        # Tune this to your expected anomaly rate (~2%)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("app_specific_audit_3.csv")
df = df.dropna(how="all")

original_df = df.copy()

# -----------------------------
# Preprocessing
# -----------------------------
df["DB_username"] = df["DB_username"].str.lower()
df["OS_username"] = df["OS_username"].str.lower()
df["Source_programname"] = df["Source_programname"].str.lower()

df = df.drop(columns=["app_specific_audit_id", "db_param_id", "mat_flg"])

df["audit_at"] = pd.to_datetime(df["audit_at"], format="%d-%m-%Y %H:%M")
df["hour"] = df["audit_at"].dt.hour
df["day_of_week"] = df["audit_at"].dt.dayofweek  # 0=Mon, 6=Sun
df = df.drop(columns=["audit_at"])

# -----------------------------
# DAY-WISE RULE BASED DETECTION
# -----------------------------
weekday_freq = df["day_of_week"].value_counts(normalize=True)
df["weekday_frequency"] = df["day_of_week"].map(weekday_freq)

# Adaptive quantile threshold
threshold = df["weekday_frequency"].quantile(DAY_ANOMALY_QUANTILE)
df["is_day_anomaly"] = (df["weekday_frequency"] < threshold).astype(int)

joblib.dump(weekday_freq.to_dict(), "weekday_freq.pkl")
joblib.dump(threshold, "weekday_threshold.pkl")

# Diagnostics: validate the threshold makes sense
print("=" * 50)
print("DAY ANOMALY DIAGNOSTICS")
print("=" * 50)
print("Weekday frequency distribution (0=Mon, 6=Sun):")
print(weekday_freq.sort_index().round(4))
print(f"\nAnomaly threshold (quantile={DAY_ANOMALY_QUANTILE}): {threshold:.4f}")
print(f"Day anomalies flagged: {df['is_day_anomaly'].sum()} / {len(df)}")
print()

# -----------------------------
# Frequency encoding
# -----------------------------
freq_maps = {}

for col in ["Src_ip", "Source_programname", "OS_username", "DB_username"]:
    freq = df[col].value_counts(normalize=True)
    freq_maps[col] = freq.to_dict()
    df[col] = df[col].map(freq)

joblib.dump(freq_maps, "freq_maps.pkl")

# -----------------------------
# Query vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer()
query_features = vectorizer.fit_transform(df["full_query"])

# -----------------------------
# Numeric feature matrix
# NOTE: day_of_week is KEPT so LOF can learn temporal patterns.
# weekday_frequency and is_day_anomaly are excluded to avoid leakage.
# -----------------------------
numeric_features = df.drop(columns=[
    "full_query",
    "full_qry_hv",
    "weekday_frequency",
    "is_day_anomaly",
    "day_of_week"
])

scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(numeric_features)
joblib.dump(numeric_features.columns.tolist(), "feature_columns.pkl")
# -----------------------------
# Combine features
# -----------------------------
X = hstack([numeric_matrix, query_features])
X_dense = X.toarray()

# -----------------------------
# Train LOF model
# -----------------------------
model = LocalOutlierFactor(
    n_neighbors=LOF_N_NEIGHBORS,
    contamination="auto",
    novelty=True
)

model.fit(X_dense)
labels = model.predict(X_dense)

# -----------------------------
# Predictions & scores
# -----------------------------
df["anomaly_flag"] = labels
df["is_anomaly"] = df["anomaly_flag"].map({1: "No", -1: "Yes"})

# LOF score (higher = more anomalous)
lof_scores = -model.negative_outlier_factor_
df["anomaly_score"] = lof_scores

min_score = lof_scores.min()
max_score = lof_scores.max()

if max_score == min_score:
    df["lof_risk"] = 0
else:
    df["lof_risk"] = 100 * (lof_scores - min_score) / (max_score - min_score)

# -----------------------------
# Query frequency
# -----------------------------
query_freq = df["full_qry_hv"].value_counts(normalize=True)
df["query_frequency"] = df["full_qry_hv"].map(query_freq)
joblib.dump(query_freq.to_dict(), "query_freq.pkl")

# -----------------------------
# FINAL RISK SCORE (HYBRID)
# -----------------------------
df["risk_score"] = df["lof_risk"]

# Rule-based boost for day anomaly
df.loc[df["is_day_anomaly"] == 1, "risk_score"] += DAY_ANOMALY_BOOST

df["risk_score"] = df["risk_score"].clip(0, 100)

# -----------------------------
# Risk level classification
# -----------------------------
df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[0, 40, 60, 80, 100],
    labels=["Safe", "Suspicious", "Risk", "High Risk"],
    include_lowest=True
)

# -----------------------------
# Anomaly reason
# -----------------------------
feature_cols = [
    "query_frequency",
    "Src_ip",
    "Source_programname",
    "OS_username",
    "hour",
    "DB_username"
]

feature_median = df[feature_cols].median()
feature_std = df[feature_cols].std()

joblib.dump(feature_median.to_dict(), "feature_median.pkl")
joblib.dump(feature_std.to_dict(), "feature_std.pkl")

reason_labels = {
    "query_frequency":    "Unusual SQL query pattern",
    "Src_ip":             "Unusual source IP behaviour",
    "Source_programname": "Unusual program access pattern",
    "OS_username":        "Unusual OS user behaviour",
    "hour":               "Unusual access time",
    "DB_username":        "Unusual DB username",
}

def get_reason(row):
    if row["anomaly_flag"] != -1 and row["is_day_anomaly"] == 0:
        return "Normal behaviour"

    reasons = []

    # ML-based reasons: flag features deviating > 2 sigma
    if row["anomaly_flag"] == -1:
        for col in feature_cols:
            if feature_std[col] == 0:
                continue
            deviation = abs(row[col] - feature_median[col]) / feature_std[col]
            if deviation > 2:
                reasons.append(reason_labels[col])

        # Fallback: if LOF flagged it but no single feature stands out clearly
        if not reasons:
            reasons.append(
                f"Model detected unusual behaviour (LOF score: {row['anomaly_score']:.2f})"
            )

    # Rule-based reason
    if row["is_day_anomaly"] == 1:
        reasons.append("Access on rare / low-activity day")

    return ", ".join(reasons)

df["anomaly_reason"] = df.apply(get_reason, axis=1)

# -----------------------------
# Merge & export
# -----------------------------
prediction_cols = df[[
    "anomaly_flag",
    "risk_score",
    "is_anomaly",
    "query_frequency",
    "anomaly_score",
    "risk_level",
    "anomaly_reason"
]]

final_df = pd.concat([original_df, prediction_cols], axis=1)
final_df.to_csv("audit_with_anomaly_results_compare.csv", index=False)

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(model, "db_anomaly_model.pkl")
joblib.dump(vectorizer, "query_vectorizer.pkl")
joblib.dump(scaler, "feature_scaler.pkl")   # Also save scaler for inference

# -----------------------------
# Summary
# -----------------------------
total = len(df)
anomaly_count = (df["anomaly_flag"] == -1).sum()
day_anomaly_count = df["is_day_anomaly"].sum()
both_anomaly = ((df["anomaly_flag"] == -1) & (df["is_day_anomaly"] == 1)).sum()

print("=" * 50)
print("ANOMALY DETECTION SUMMARY")
print("=" * 50)
print(f"  Total records        : {total}")
print(f"  ML anomalies (LOF)   : {anomaly_count} ({anomaly_count / total:.1%})")
print(f"  Day anomalies (rule) : {day_anomaly_count} ({day_anomaly_count / total:.1%})")
print(f"  Both flagged         : {both_anomaly}")
print(f"  Normal               : {total - anomaly_count}")
print(f"  Contamination param  : {LOF_CONTAMINATION:.1%}  ← tune if ML anomaly % looks off")
print()
print("Risk level breakdown:")
print(df["risk_level"].value_counts().sort_index())
print()
print("Outputs saved:")
print("  audit_with_anomaly_results_compare.csv")
print("  db_anomaly_model.pkl")
print("  query_vectorizer.pkl")
print("  feature_scaler.pkl")


# def predict_new_data(input_df: pd.DataFrame):
#     df = input_df.copy()

#     # -----------------------------
#     # Preprocessing
#     # -----------------------------
#     df["DB_username"] = df["DB_username"].str.lower()
#     df["OS_username"] = df["OS_username"].str.lower()
#     df["Source_programname"] = df["Source_programname"].str.lower()

#     df["audit_at"] = pd.to_datetime(df["audit_at"], format="%d-%m-%Y %H:%M")
#     df["hour"] = df["audit_at"].dt.hour
#     df["day_of_week"] = df["audit_at"].dt.dayofweek
#     df = df.drop(columns=["audit_at"])

#     # -----------------------------
#     # Load saved stats
#     # -----------------------------
#     weekday_freq = joblib.load("weekday_freq.pkl")
#     threshold = joblib.load("weekday_threshold.pkl")
#     freq_maps = joblib.load("freq_maps.pkl")
#     feature_cols = joblib.load("feature_columns.pkl")
#     feature_median = joblib.load("feature_median.pkl")
#     feature_std = joblib.load("feature_std.pkl")

#     # -----------------------------
#     # Day anomaly
#     # -----------------------------
#     df["weekday_frequency"] = df["day_of_week"].map(weekday_freq).fillna(0)
#     df["is_day_anomaly"] = (df["weekday_frequency"] < threshold).astype(int)

#     query_freq_map = joblib.load("query_freq.pkl")

#     df["query_frequency"] = df["full_qry_hv"].map(query_freq_map).fillna(0)

#     # -----------------------------
#     # Frequency encoding
#     # -----------------------------
#     for col in ["Src_ip", "Source_programname", "OS_username", "DB_username"]:
#         df[col] = df[col].map(freq_maps[col]).fillna(0)

#     # -----------------------------
#     # Load model artifacts
#     # -----------------------------
#     model = joblib.load("db_anomaly_model.pkl")
#     vectorizer = joblib.load("query_vectorizer.pkl")
#     scaler = joblib.load("feature_scaler.pkl")

#     # -----------------------------
#     # Transform
#     # -----------------------------
#     query_features = vectorizer.transform(df["full_query"])

#     for col in feature_cols:
#         if col not in df:
#             df[col] = 0

#     numeric_features = df[feature_cols]
#     numeric_matrix = scaler.transform(numeric_features)

#     X = hstack([numeric_matrix, query_features])
#     X_dense = X.toarray()

#     # -----------------------------
#     # Predict
#     # -----------------------------
#     labels = model.predict(X_dense)
#     lof_scores = -model.decision_function(X_dense)

#     # -----------------------------
#     # Reason logic
#     # -----------------------------
#     feature_cols_reason = [
#         "query_frequency",
#         "Src_ip",
#         "Source_programname",
#         "OS_username",
#         "hour",
#         "DB_username"
#     ]

#     reason_labels = {
#         "query_frequency":    "Unusual SQL query pattern",
#         "Src_ip":             "Unusual source IP behaviour",
#         "Source_programname": "Unusual program access pattern",
#         "OS_username":        "Unusual OS user behaviour",
#         "hour":               "Unusual access time",
#         "DB_username":        "Unusual DB username",
#     }

#     reasons = []

#     # ML-based reason
#     if labels[0] == -1:
#         for col in feature_cols_reason:
#             if feature_std[col] == 0:
#                 continue

#             val = df[col].iloc[0]
#             median = feature_median[col]
#             std = feature_std[col]

#             deviation = abs(val - median) / std

#             if deviation > 2:
#                 reasons.append(reason_labels[col])

#         if not reasons:
#             reasons.append(f"Model detected unusual behaviour (LOF score: {lof_scores[0]:.2f})")

#     # Rule-based reason
#     if df["is_day_anomaly"].iloc[0] == 1:
#         reasons.append("Access on rare / low-activity day")

#     if not reasons:
#         reasons = ["Normal behaviour"]

#     # -----------------------------
#     # Final output
#     # -----------------------------
#     return {
#         "anomaly_flag": int(labels[0]),
#         "lof_score": float(lof_scores[0]),
#         "reason": ", ".join(reasons)
#     }