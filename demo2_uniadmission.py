import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path

# -----------------------------
# 1) Load data
# -----------------------------
CSV_PATH = "sample_data_v2.csv"  # از فایل جدید استفاده کن
df = pd.read_csv(CSV_PATH)

# Target: Major_code = 1 if {پزشکی, دندانپزشکی, داروسازی} else 0
top_majors = {"پزشکی", "دندانپزشکی", "داروسازی"}
df["Major_code"] = df["Major"].apply(lambda x: 1 if x in top_majors else 0)

# Features (ساده و بدون لو دادن پیش‌پردازش‌های اصلی)
feature_cols = [
    "Year",
    "General.PersianLiterature","General.Arabic","General.IslamicStudies","General.English",
    "ExpScience.Geology","ExpScience.Math","ExpScience.Biology","ExpScience.Physics","ExpScience.Chemistry",
    "Rank.in.Total","Quota","Rank.in.Quota"
]
X = df[feature_cols].copy()
y = df["Major_code"].values

# -----------------------------
# 2) Cross-Validation setup
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

fold_accs = []
frames = []

# -----------------------------
# 3) CV loop (frame per fold)
# -----------------------------
fold = 1
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    fold_accs.append(acc)

    # Scatter: احتمال قبولی برای نمونه‌های فولد
    plt.figure(figsize=(6,4))
    # رنگ: براساس y_test (برچسب واقعی) / دور خط: نمونه‌های خطا
    errors = (y_pred != y_test)
    plt.scatter(range(len(y_proba)), y_proba, c=y_test, cmap="coolwarm", edgecolors=np.where(errors, "k", "none"))
    plt.title(f"Fold {fold} — Probabilities (acc={acc:.2f})")
    plt.xlabel("Sample index in fold")
    plt.ylabel("P(Top Major)")
    plt.ylim(-0.05, 1.05)
    png_path = f"cv_fold_{fold}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()

    frames.append(imageio.imread(png_path))
    fold += 1

# -----------------------------
# 4) Save GIF + print metrics
# -----------------------------
gif_name = "rf_cv_demo.gif"
imageio.mimsave(gif_name, frames, duration=1.2)

print("Per-fold accuracy:", [f"{a:.3f}" for a in fold_accs])
print("Mean accuracy:", np.mean(fold_accs).round(3))

# گزارش کلی با یک بار fit روی کل داده (صرفاً برای نمایش)
clf.fit(X, y)
y_pred_all = clf.predict(X)
print("\nClassification report (fit on full data, for display only):")
print(classification_report(y, y_pred_all))
print(f"\nDemo GIF saved as {gif_name}")
