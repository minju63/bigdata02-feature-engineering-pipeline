"""STEP 03-05: 전처리 조합 실험 (Base/Exp-1~3 + 전체 그리드 + Drop NA).
평가: 학습셋(80%) 위 RepeatedStratifiedKFold(5x2) CV  +  hold-out 테스트셋(20%)."""
import itertools, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from common import MODELS, OUT, SEED, build_pipeline, load_data

os.makedirs(OUT, exist_ok=True)
df = load_data()
y = df["Survived"]; X = df.drop(columns="Survived")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
SCORING = {"acc": "accuracy", "prec": "precision", "rec": "recall", "f1": "f1", "auc": "roc_auc"}


def run(name, model, cfg):
    """cfg: impute, encode, scale, select, use_fe, base, drop"""
    c = dict(cfg)
    drop = c.pop("drop", False)
    a, b, ya, yb = Xtr, Xte, ytr, yte
    if drop:   # Drop NA: Age/Embarked 결측 행 제거 (학습·평가 모두)
        ma = Xtr[["Age", "Embarked"]].notna().all(axis=1); mb = Xte[["Age", "Embarked"]].notna().all(axis=1)
        a, ya, b, yb = Xtr[ma], ytr[ma], Xte[mb], yte[mb]
    pipe = build_pipeline(model, **c)
    cv = cross_validate(pipe, a, ya, cv=CV, scoring=SCORING, return_train_score=False, n_jobs=1)
    pipe.fit(a, ya)
    p = pipe.predict(b); pr = pipe.predict_proba(b)[:, 1]
    r = dict(exp=name, model=model, n_train=len(a), n_test=len(b), **cfg)
    for k in SCORING:
        r[f"cv_{k}"] = cv[f"test_{k}"].mean(); r[f"cv_{k}_std"] = cv[f"test_{k}"].std()
    r.update(test_acc=accuracy_score(yb, p), test_prec=precision_score(yb, p), test_rec=recall_score(yb, p),
             test_f1=f1_score(yb, p), test_auc=roc_auc_score(yb, pr),
             train_acc=accuracy_score(ya, pipe.predict(a)))
    r["gap"] = r["train_acc"] - r["cv_acc"]      # 과적합 지표: 학습 정확도 - CV 정확도
    return r


def main():
    t0 = time.time()
    jobs = []
    # ---- (A) 과제 필수 실험표: Base, Exp-1~3  (FE 없음 / FE 포함 두 버전)
    EXPS = {
        "Base":  dict(base=True),
        "Exp-1": dict(impute="mean", encode="onehot", scale="standard", select=False),
        "Exp-2": dict(impute="median", encode="label", scale="minmax", select=True),
        "Exp-3": dict(impute="most_frequent", encode="onehot", scale="robust", select=True),
    }
    for m in MODELS:
        jobs.append(("A", "Base", m, dict(base=True)))
        for e, cfg in list(EXPS.items())[1:]:
            jobs.append(("A", e, m, {**cfg, "use_fe": False}))
            jobs.append(("A", e + "+FE", m, {**cfg, "use_fe": True}))
    # ---- (B) 전체 그리드
    for imp, enc, sc, sel, fe in itertools.product(
            ["mean", "median", "most_frequent"], ["onehot", "label"],
            ["standard", "minmax", "robust", "none"], [False, True], [False, True]):
        for m in MODELS:
            jobs.append(("B", "grid", m, dict(impute=imp, encode=enc, scale=sc, select=sel, use_fe=fe)))
    # ---- (C) 결측치 처리 비교(Drop NA 포함): onehot / standard / FS 없음 / FE 포함
    for imp in ["mean", "median", "most_frequent", "dropna"]:
        for m in MODELS:
            cfg = dict(impute="median" if imp == "dropna" else imp, encode="onehot", scale="standard",
                       select=False, use_fe=True, drop=(imp == "dropna"))
            jobs.append(("C", imp, m, cfg))
    print(len(jobs), "jobs", flush=True)
    res = Parallel(n_jobs=-1, verbose=5)(delayed(run)(n, m, c) for (_, n, m, c) in jobs)
    out = pd.DataFrame(res); out["group"] = [j[0] for j in jobs]
    out.to_csv(f"{OUT}/all_results.csv", index=False)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
