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
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for a, c, b in zip(ax, ["Age", "Fare", "SibSp", "Parch"], [30, 40, 9, 7]):
    sns.histplot(df[c].dropna(), bins=b, kde=(c in ("Age", "Fare")), ax=a, color="#4C72B0")
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

# ---- 그림 6: Age 결측은 무작위인가? (결측 여부별 생존율 / 객실 등급 구성 + 카이제곱 검정)
from scipy.stats import chi2_contingency
miss = df["Age"].isna().map({False: "Age 있음", True: "Age 결측"})
p_surv = chi2_contingency(pd.crosstab(miss, df["Survived"]))[1]
p_cls = chi2_contingency(pd.crosstab(miss, df["Pclass"]))[1]
S["age_missing_test"] = {"p_survival": float(p_surv), "p_pclass": float(p_cls),
                         "pclass3_share_missing": round(float((df.loc[df.Age.isna(), "Pclass"] == 3).mean()), 3),
                         "pclass3_share_present": round(float((df.loc[df.Age.notna(), "Pclass"] == 3).mean()), 3)}
json.dump(S, open(f"{OUT}/eda_summary.json", "w"), ensure_ascii=False, indent=2)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
order = ["Age 있음", "Age 결측"]
g = df.groupby(miss)["Survived"].agg(["mean", "count"]).loc[order]
sns.barplot(x=g.index, y=g["mean"], ax=ax[0], color="#4C72B0", order=order)
for i, (m_, n_) in enumerate(zip(g["mean"], g["count"])):
    ax[0].text(i, m_ + 0.02, f"{m_:.3f} (n={n_})", ha="center")
ax[0].set_ylim(0, 0.8); ax[0].set_ylabel("생존율"); ax[0].set_xlabel("")
ax[0].set_title(f"Age 결측 여부별 생존율 (카이제곱 p={p_surv:.3f})")
ct = pd.crosstab(miss, df["Pclass"], normalize="index").loc[order]
ct.plot.bar(stacked=True, ax=ax[1], color=["#55A868", "#DD8452", "#C44E52"], rot=0)
ax[1].set_title(f"Age 결측 여부별 객실 등급 구성 (p={p_cls:.1e})"); ax[1].set_xlabel("")
ax[1].set_ylabel("비율"); ax[1].legend(title="Pclass", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/eda_age_missing.png"); plt.close()

# ---- 그림 7: HasCabin(객실 번호 기록 여부)과 생존
hc = df["Cabin"].notna().map({False: "Cabin 없음(0)", True: "Cabin 있음(1)"})
p_hc = chi2_contingency(pd.crosstab(hc, df["Survived"]))[1]
S["hascabin"] = {"n1": int(df["Cabin"].notna().sum()), "n0": int(df["Cabin"].isna().sum()),
                 "surv1": round(float(df.loc[df.Cabin.notna(), "Survived"].mean()), 3),
                 "surv0": round(float(df.loc[df.Cabin.isna(), "Survived"].mean()), 3), "p": float(p_hc),
                 "pclass1_share_has": round(float((df.loc[df.Cabin.notna(), "Pclass"] == 1).mean()), 3)}
json.dump(S, open(f"{OUT}/eda_summary.json", "w"), ensure_ascii=False, indent=2)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
o2 = ["Cabin 없음(0)", "Cabin 있음(1)"]
g = df.groupby(hc)["Survived"].agg(["mean", "count"]).loc[o2]
sns.barplot(x=g.index, y=g["mean"], ax=ax[0], color="#4C72B0", order=o2)
for i, (m_, n_) in enumerate(zip(g["mean"], g["count"])):
    ax[0].text(i, m_ + 0.02, f"{m_:.3f} (n={n_})", ha="center")
ax[0].set_ylim(0, 0.85); ax[0].set_ylabel("생존율"); ax[0].set_xlabel("")
ax[0].set_title(f"HasCabin별 생존율 (카이제곱 p={p_hc:.0e})")
d2 = df.assign(HasCabin=hc).groupby(["Pclass", "HasCabin"])["Survived"].mean().reset_index()
sns.barplot(data=d2, x="Pclass", y="Survived", hue="HasCabin", hue_order=o2, ax=ax[1], palette=["#999999", "#4C72B0"])
ax[1].set_ylim(0, 1); ax[1].set_ylabel("생존율"); ax[1].set_title("객실 등급별 HasCabin 생존율"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/eda_hascabin.png"); plt.close()

# ---- 그림 8: 범주형 변수 분포 (Countplot)
S["cat_share"] = {c: (df[c].value_counts(normalize=True) * 100).round(1).to_dict() for c in ["Sex", "Pclass", "Embarked"]}
json.dump(S, open(f"{OUT}/eda_summary.json", "w"), ensure_ascii=False, indent=2)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, c in zip(ax, ["Sex", "Pclass", "Embarked"]):
    sns.countplot(data=df, x=c, ax=a, color="#4C72B0")
    for p_ in a.patches:
        a.annotate(f"{int(p_.get_height())}", (p_.get_x() + p_.get_width() / 2, p_.get_height()), ha="center", va="bottom")
    a.set_title(f"{c} 분포"); a.set_ylabel("인원 수"); a.set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_count.png"); plt.close()

print(json.dumps(S, ensure_ascii=False, indent=1))
