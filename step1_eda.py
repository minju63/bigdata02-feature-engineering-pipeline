"""STEP 01-02: 데이터 준비 및 EDA. 그림은 outputs/ 에 저장, 요약 수치는 outputs/eda_summary.json."""
import json, os
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from common import load_data, OUT

os.makedirs(OUT, exist_ok=True)
df = load_data()
S = {}
S["shape"] = list(df.shape)
S["dtypes"] = df.dtypes.astype(str).to_dict()
S["missing"] = (df.isna().mean() * 100).round(2).to_dict()
S["missing_n"] = df.isna().sum().to_dict()
S["target"] = df["Survived"].value_counts().to_dict()
S["target_ratio"] = df["Survived"].value_counts(normalize=True).round(4).to_dict()
S["duplicates"] = int(df.duplicated().sum())
S["describe"] = df[["Age", "SibSp", "Parch", "Fare"]].describe().round(2).to_dict()
S["surv_by_sex"] = df.groupby("Sex")["Survived"].mean().round(3).to_dict()
S["surv_by_pclass"] = df.groupby("Pclass")["Survived"].mean().round(3).to_dict()
S["surv_by_embarked"] = df.groupby("Embarked")["Survived"].mean().round(3).to_dict()

# 이상치 (IQR)
out = {}
for c in ["Age", "SibSp", "Parch", "Fare"]:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out[c] = int(((df[c] < lo) | (df[c] > hi)).sum())
S["outliers_iqr"] = out
S["fare_skew"] = round(float(df["Fare"].skew()), 2)
S["age_skew"] = round(float(df["Age"].skew()), 2)
S["fare_zero"] = int((df["Fare"] == 0).sum())

num = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]
corr = num.corr()
S["corr_with_target"] = corr["Survived"].drop("Survived").round(3).to_dict()
# Age 결측 vs 생존
S["surv_age_missing"] = df.groupby(df["Age"].isna())["Survived"].mean().round(3).to_dict()
json.dump(S, open(f"{OUT}/eda_summary.json", "w"), ensure_ascii=False, indent=2)

# ---- 그림 1: 결측치 + 타깃 분포 (Barplot / Countplot)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
m = (df.isna().mean() * 100).sort_values(ascending=False)
m = m[m > 0]
sns.barplot(x=m.values, y=m.index, ax=ax[0], color="#4C72B0")
for i, v in enumerate(m.values):
    ax[0].text(v + 1, i, f"{v:.1f}%", va="center")
ax[0].set_xlim(0, 100); ax[0].set_title("결측치 비율 (%)"); ax[0].set_xlabel("")
sns.countplot(x="Survived", data=df, ax=ax[1], palette=["#C44E52", "#55A868"], hue="Survived", legend=False)
for p in ax[1].patches:
    ax[1].annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha="center", va="bottom")
ax[1].set_xticklabels(["사망(0)", "생존(1)"]); ax[1].set_title("타깃 변수 분포 (Survived)"); ax[1].set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_missing_target.png"); plt.close()

# ---- 그림 2: 히스토그램
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
for a, c, b in zip(ax, ["Age", "Fare", "SibSp"], [30, 40, 9]):
    sns.histplot(df[c].dropna(), bins=b, kde=(c != "SibSp"), ax=a, color="#4C72B0")
    a.set_title(f"{c} 분포 (왜도 {df[c].skew():.2f})")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_hist.png"); plt.close()

# ---- 그림 3: 박스플롯
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, c in zip(ax, ["Age", "Fare", "SibSp", "Parch"]):
    sns.boxplot(data=df, x="Survived", y=c, ax=a, hue="Survived", palette=["#C44E52", "#55A868"], legend=False)
    a.set_title(f"{c} (이상치 {out[c]}개)")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_box.png"); plt.close()

# ---- 그림 4: 히트맵
plt.figure(figsize=(5.5, 4.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("수치형 변수 상관관계"); plt.tight_layout(); plt.savefig(f"{OUT}/eda_heatmap.png"); plt.close()

# ---- 그림 5: 범주형 vs 생존
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, c in zip(ax, ["Sex", "Pclass", "Embarked"]):
    sns.barplot(data=df, x=c, y="Survived", ax=a, color="#4C72B0", errorbar=None)
    for p in a.patches:
        a.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha="center", va="bottom")
    a.set_ylim(0, 1); a.set_title(f"{c}별 생존율"); a.set_ylabel("생존율")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_cat_survival.png"); plt.close()

print(json.dumps(S, ensure_ascii=False, indent=1))
