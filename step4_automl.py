"""가산점: AutoML(FLAML) 비교 실험.
설정 A: 원본 7개 변수 그대로 (결측치·범주형 처리도 AutoML에 맡김)
설정 B: 파생변수 6개를 포함한 13개 변수 (FeatureEngineer 출력, 나머지 전처리는 AutoML에 맡김)
동일한 학습/테스트 분할 사용. 탐색 시간은 시간 예산(초)으로 제한."""
import json, time
import numpy as np, pandas as pd
from flaml import AutoML
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from common import FeatureEngineer, OUT, SEED, load_data

BUDGET = 60
df = load_data(); y = df["Survived"]; X = df.drop(columns="Survived")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def to_cat(a):
    a = a.copy()
    for c in a.columns:
        if a[c].dtype == object or str(a[c].dtype).startswith("str") or c in ("Embarked", "Sex", "Title", "AgeGroup"):
            a[c] = a[c].astype("category")
    return a


raw_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
sets = {
    "A: 원본 7개 변수": (to_cat(Xtr[raw_cols]), to_cat(Xte[raw_cols])),
    "B: 파생변수 포함 13개 변수": (to_cat(FeatureEngineer(True).transform(Xtr)), to_cat(FeatureEngineer(True).transform(Xte))),
}
rows = []
for name, (a, b) in sets.items():
    t0 = time.time()
    am = AutoML()
    am.fit(a, ytr, task="classification", metric="accuracy", time_budget=BUDGET, eval_method="cv", n_splits=5,
           estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"], seed=SEED, verbose=0, n_jobs=1)
    p = am.predict(b); pr = am.predict_proba(b)[:, 1]
    # AutoML이 고른 최종 모델을 우리와 같은 교차검증(5-fold x 2, 학습셋)으로 재평가
    our_cv = cross_val_score(clone(am.model.estimator), a, ytr, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED), scoring='accuracy').mean()
    rows.append(dict(setting=name, best_estimator=am.best_estimator, n_features=a.shape[1], search_s=round(time.time() - t0),
                     automl_cv_acc=1 - am.best_loss, our_cv_acc=our_cv, test_acc=accuracy_score(yte, p), test_prec=precision_score(yte, p),
                     test_rec=recall_score(yte, p), test_f1=f1_score(yte, p), test_auc=roc_auc_score(yte, pr)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/automl_results.csv", index=False)
