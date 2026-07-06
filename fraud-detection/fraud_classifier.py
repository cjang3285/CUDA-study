"""
Supervised layer for the fraud cases that unsupervised graph methods (Louvain
community detection, LOF, personalized PageRank) could not separate from normal
transactions. Trains GPU XGBoost on V1..V28 + Amount + Time, with and without the
unsupervised kNN-distance ("how locally anomalous is this point") feature, and
reports PR-AUC plus how many of the specific "missed by Louvain" frauds it recovers.
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

df = pd.read_csv("data/creditcard.csv")

anomaly = pd.read_csv("results/node_groups_with_anomaly_score.csv")[
    ["row_id", "mean_knn_dist", "partition"]
]
df["row_id"] = np.arange(len(df))
df = df.merge(anomaly, on="row_id", how="left")

top = df.groupby("partition")["Class"].agg(["count", "sum"])
top["rate"] = top["sum"] / top["count"]
top = top[top["count"] >= 50].sort_values("rate", ascending=False)
comm_a, comm_b = int(top.index[0]), int(top.index[1])
df["is_missed_fraud"] = (df["Class"] == 1) & (~df["partition"].isin([comm_a, comm_b]))

train_idx, test_idx = train_test_split(
    df.index, test_size=0.3, stratify=df["Class"], random_state=42
)
train, test = df.loc[train_idx], df.loc[test_idx]

scale_pos_weight = (train["Class"] == 0).sum() / (train["Class"] == 1).sum()
print(f"train size={len(train)}, test size={len(test)}, scale_pos_weight={scale_pos_weight:.1f}")


def run(feature_set, name):
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(train[feature_set], train["Class"])
    proba = model.predict_proba(test[feature_set])[:, 1]
    pr_auc = average_precision_score(test["Class"], proba)

    test_scored = test.copy()
    test_scored["proba"] = proba
    missed_in_test = test_scored[test_scored["is_missed_fraud"]]
    normal_in_test = test_scored[test_scored["Class"] == 0]

    print(f"\n=== {name} ({time.time()-t0:.1f}s) ===")
    print(f"features: {feature_set}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"missed-fraud proba: mean={missed_in_test['proba'].mean():.3f}, "
          f"median={missed_in_test['proba'].median():.3f}")
    print(f"normal proba:       mean={normal_in_test['proba'].mean():.4f}, "
          f"median={normal_in_test['proba'].median():.4f}")

    n_fraud_total = (test["Class"] == 1).sum()
    precision, recall, thresholds = precision_recall_curve(test["Class"], proba)
    for target_recall in [0.5, 0.7, 0.8, 0.9]:
        i = np.searchsorted(recall[::-1], target_recall)
        i = len(recall) - 1 - i
        i = max(0, min(i, len(precision) - 1))
        thresh = thresholds[min(i, len(thresholds) - 1)]
        flagged = test_scored[test_scored["proba"] >= thresh]
        tp = (flagged["Class"] == 1).sum()
        fp = (flagged["Class"] == 0).sum()
        print(f"  at recall>={target_recall}: precision={precision[i]:.3f} -> "
              f"flag {len(flagged)} txns ({tp}/{n_fraud_total} frauds caught, "
              f"{fp} normal txns wrongly flagged, {n_fraud_total - tp} frauds missed)")

    n_missed = len(missed_in_test)
    hits = (missed_in_test["proba"] > 0.5).sum()
    print(f"of {n_missed} 'Louvain-missed' frauds in test set, "
          f"{hits} ({hits/n_missed*100:.0f}%) score > 0.5 with this classifier")

    # what threshold, precision, and false-positive count does catching EVERY
    # single fraud (recall = 1.0) actually require?
    min_fraud_score = test_scored.loc[test_scored["Class"] == 1, "proba"].min()
    flagged = test_scored[test_scored["proba"] >= min_fraud_score]
    n_fraud_total = (test["Class"] == 1).sum()
    fp = (flagged["Class"] == 0).sum()
    precision_at_full_recall = n_fraud_total / len(flagged)
    print(f"to catch ALL {n_fraud_total} frauds (recall=1.0): must flag {len(flagged)} "
          f"transactions ({fp} of them normal) -> precision={precision_at_full_recall:.5f} "
          f"({fp} false alarms per real fraud caught)")

    return pr_auc, model


pr_auc_baseline, base_model = run(feature_cols, "baseline: V1-28 + Amount + Time")
pr_auc_augmented, _ = run(feature_cols + ["mean_knn_dist"], "augmented: + graph kNN-distance feature")

# hand-engineered features: time-of-day, log(Amount), and pairwise interactions of
# the model's own top-5 most important raw features
importances = pd.Series(base_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top5 = importances.head(5).index.tolist()
print(f"\ntop-5 most important raw features: {top5}")

df["hour_of_day"] = (df["Time"] % 86400) // 3600
df["log_amount"] = np.log1p(df["Amount"])
engineered_cols = ["hour_of_day", "log_amount"]
for i in range(len(top5)):
    for j in range(i + 1, len(top5)):
        col = f"{top5[i]}_x_{top5[j]}"
        df[col] = df[top5[i]] * df[top5[j]]
        engineered_cols.append(col)

train, test = df.loc[train_idx], df.loc[test_idx]
pr_auc_engineered, _ = run(feature_cols + engineered_cols, "engineered: + hour/log(Amount)/top-5 interactions")

print(f"\nPR-AUC baseline={pr_auc_baseline:.4f}, "
      f"graph-augmented={pr_auc_augmented:.4f}, "
      f"engineered={pr_auc_engineered:.4f}")
