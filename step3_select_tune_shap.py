"""STEP 04 변수 선택(RF importance, top-N sweep) + 가산점(GridSearchCV, SHAP) + 요약 그림."""
import json
import numpy as np, pandas as pd, seaborn as sns, shap
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier
from common import MODELS, OUT, SEED, build_pipeline, feature_names, load_data

df = load_data(); y = df["Survived"]; X = df.drop(columns="Survived")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
BASECFG = dict(impute="mean", encode="onehot", scale="standard", use_fe=True)
S = {}

# ---- (1) RF 중요도 (전체 변수)
pipe = build_pipeline("RandomForest", **BASECFG, select=False).fit(Xtr, ytr)
names = feature_names(pipe)
imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values()
plt.figure(figsize=(6.5, 7)); imp.plot.barh(color="#4C72B0")
plt.title("Random Forest Feature Importance (전체 %d개 변수)" % len(imp)); plt.tight_layout()
plt.savefig(f"{OUT}/fs_importance.png"); plt.close()
imp.sort_values(ascending=False).to_csv(f"{OUT}/fs_importance.csv")
S["top8"] = list(imp.sort_values(ascending=False).index[:8])
S["n_features_all"] = len(imp)


# ---- (2) top-N sweep: 제거 전/후 성능
def sweep(model, n):
    sel = n is not None
    p = build_pipeline(model, **BASECFG, select=sel, top_n=n or 0)
    cv = cross_validate(p, Xtr, ytr, cv=CV, scoring={"acc": "accuracy", "auc": "roc_auc"})
    p.fit(Xtr, ytr)
    return dict(model=model, top_n=n if sel else len(imp), cv_acc=cv["test_acc"].mean(), cv_auc=cv["test_auc"].mean(),
                train_acc=accuracy_score(ytr, p.predict(Xtr)), test_acc=accuracy_score(yte, p.predict(Xte)),
                test_auc=roc_auc_score(yte, p.predict_proba(Xte)[:, 1]))


jobs = [(m, n) for m in MODELS for n in [3, 5, 8, 12, 16, None]]
sw = pd.DataFrame(Parallel(n_jobs=-1)(delayed(sweep)(m, n) for m, n in jobs))
sw["gap"] = sw.train_acc - sw.cv_acc
sw.to_csv(f"{OUT}/fs_sweep.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for m, g in sw.groupby("model"):
    g = g.sort_values("top_n")
    ax[0].plot(g.top_n, g.cv_acc, "o-", label=m); ax[1].plot(g.top_n, g.gap, "o-", label=m)
ax[0].set_title("선택 변수 수(N)별 CV 정확도"); ax[1].set_title("과적합 지표 (학습정확도 − CV정확도)")
for a in ax: a.set_xlabel("상위 N개 변수 (%d = 전체)" % len(imp)); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fs_sweep.png"); plt.close()

# ---- (3) GridSearchCV (가산점) : XGBoost / RandomForest, 파이프라인 전체를 탐색
tuned = {}
grids = {
    "XGBoost": {"model__n_estimators": [100, 300], "model__max_depth": [2, 3, 4, 6],
                "model__learning_rate": [0.03, 0.1], "model__subsample": [0.8, 1.0]},
    "RandomForest": {"model__n_estimators": [200, 500], "model__max_depth": [4, 6, 8, None],
                     "model__min_samples_leaf": [1, 3, 5]},
}
for m, g in grids.items():
    p = build_pipeline(m, **BASECFG, select=False)
    gs = GridSearchCV(p, g, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
    pr = gs.predict(Xte); pp = gs.predict_proba(Xte)[:, 1]
    d0 = build_pipeline(m, **BASECFG, select=False).fit(Xtr, ytr)
    tuned[m] = dict(best_params={k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                    cv_acc=gs.best_score_, test_acc=accuracy_score(yte, pr), test_f1=f1_score(yte, pr),
                    test_auc=roc_auc_score(yte, pp), default_test_acc=accuracy_score(yte, d0.predict(Xte)),
                    default_test_auc=roc_auc_score(yte, d0.predict_proba(Xte)[:, 1]))
    if m == "XGBoost":
        best_xgb = gs.best_estimator_
S["gridsearch"] = tuned

# ---- (4) SHAP (튜닝된 XGBoost)
pre = best_xgb[:-1]
Xt = pd.DataFrame(pre.transform(Xte), columns=feature_names(best_xgb))
expl = shap.TreeExplainer(best_xgb.named_steps["model"])
sv = expl.shap_values(Xt)
plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=12)
plt.title("SHAP Summary (튜닝된 XGBoost, 테스트셋)"); plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary.png", bbox_inches="tight"); plt.close()
plt.figure(); shap.summary_plot(sv, Xt, plot_type="bar", show=False, max_display=12)
plt.tight_layout(); plt.savefig(f"{OUT}/shap_bar.png", bbox_inches="tight"); plt.close()
S["shap_top"] = list(pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False).head(8).round(3).items())

# ---- (5) 요약 그림 (all_results.csv)
d = pd.read_csv(f"{OUT}/all_results.csv")
a = d[d.group == "A"]
order = ["Base", "Exp-1", "Exp-1+FE", "Exp-2", "Exp-2+FE", "Exp-3", "Exp-3+FE"]
hm = a.pivot(index="exp", columns="model", values="cv_acc").loc[order]
plt.figure(figsize=(6.5, 4)); sns.heatmap(hm, annot=True, fmt=".3f", cmap="YlGn", vmin=0.69, vmax=0.84)
plt.title("실험별 CV 정확도 (5-fold x 2)"); plt.ylabel(""); plt.xlabel(""); plt.tight_layout()
plt.savefig(f"{OUT}/exp_heatmap.png"); plt.close()

b = d[d.group == "B"]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for a_, f, t in zip(ax, ["use_fe", "encode", "scale", "select"], ["파생변수(FE)", "인코딩", "스케일링", "변수선택(Top-8)"]):
    pv = b.pivot_table(index=f, columns="model", values="cv_acc")
    pv.T.plot.bar(ax=a_, width=0.8, legend=True); a_.set_title(t); a_.set_ylim(0.78, 0.84); a_.set_xlabel("")
    a_.tick_params(axis="x", rotation=30); a_.legend(fontsize=7)
ax[0].set_ylabel("CV 정확도(그리드 평균)")
plt.tight_layout(); plt.savefig(f"{OUT}/marginal_effects.png"); plt.close()

# 모델별 요약 수치 (보고서용)
S["marginal"] = {f: b.pivot_table(index=f, columns="model", values="cv_acc").round(4).to_dict() for f in
                 ["use_fe", "encode", "scale", "select", "impute"]}
S["gap_select"] = b.pivot_table(index="select", columns="model", values="gap").round(4).to_dict()
S["best_overall"] = b.sort_values("cv_acc", ascending=False).iloc[0].to_dict()
json.dump(S, open(f"{OUT}/step3_summary.json", "w"), ensure_ascii=False, indent=2, default=str)
print(json.dumps(S, ensure_ascii=False, indent=1, default=str)[:4000])
print(sw.round(3).to_string())
