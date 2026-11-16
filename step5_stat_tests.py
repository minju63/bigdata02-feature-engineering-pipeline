"""요인별 차이의 통계 검정: 같은 fold 분할에서 기준 설정과 한 요인만 바꾼 설정을 쌍으로 비교.
기준 설정 = Exp-1+FE (Mean, One-Hot, Standard, 변수 선택 없음, 파생변수 사용).
검정 = 보정된 resampled paired t-test (Nadeau & Bengio, 2003): 반복 CV의 fold 점수는 서로 겹치는 학습 데이터를
공유해 독립이 아니므로, 분산에 (1/k + n_test/n_train) 보정을 적용한다. 지표는 정확도(accuracy)."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from common import MODELS, OUT, SEED, build_pipeline, load_data

df = load_data(); y = df["Survived"]; X = df.drop(columns="Survived")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)   # 모든 설정이 같은 분할 사용
K, RATIO = 10, 1 / 4                                                      # 총 fold 수, n_test/n_train
BASE = dict(impute="mean", encode="onehot", scale="standard", select=False, use_fe=True)
VARIANTS = {
    "파생변수 없음 (vs 있음)": dict(use_fe=False),
    "Label 인코딩 (vs One-Hot)": dict(encode="label"),
    "스케일링 MinMax (vs Standard)": dict(scale="minmax"),
    "스케일링 Robust (vs Standard)": dict(scale="robust"),
    "스케일링 없음 (vs Standard)": dict(scale="none"),
    "변수 선택 Top-8 (vs 안 함)": dict(select=True),
    "대치 Median (vs Mean)": dict(impute="median"),
    "대치 Most Frequent (vs Mean)": dict(impute="most_frequent"),
}


def scores(m, cfg):
    return cross_val_score(build_pipeline(m, **cfg), Xtr, ytr, cv=CV, scoring="accuracy")


def one(m):
    b = scores(m, BASE); out = []
    for name, ch in VARIANTS.items():
        v = scores(m, {**BASE, **ch}); d = v - b
        sd = d.std(ddof=1)
        t = d.mean() / np.sqrt((1 / K + RATIO) * sd ** 2) if sd > 0 else 0.0
        p = 2 * stats.t.sf(abs(t), K - 1) if sd > 0 else 1.0
        out.append(dict(model=m, comparison=name, base_acc=b.mean(), variant_acc=v.mean(), diff=d.mean(), t=t, p=p))
    # 전처리 전체 효과: Base(수치형 4개만) vs 기준 설정
    bb = cross_val_score(build_pipeline(m, base=True), Xtr, ytr, cv=CV, scoring="accuracy"); d = b - bb
    t = d.mean() / np.sqrt((1 / K + RATIO) * d.std(ddof=1) ** 2)
    out.append(dict(model=m, comparison="Base -> Exp-1+FE", base_acc=bb.mean(), variant_acc=b.mean(), diff=d.mean(), t=t, p=2 * stats.t.sf(abs(t), K - 1)))
    return out


res = pd.DataFrame([r for rr in Parallel(n_jobs=-1)(delayed(one)(m) for m in MODELS) for r in rr])
res.to_csv(f"{OUT}/stat_tests.csv", index=False)
pd.set_option("display.width", 200)
res["sig"] = np.where(res.p < 0.05, "*", "")
print(res.assign(diff=(res["diff"] * 100).round(2), p=res.p.round(3), base_acc=res.base_acc.round(3), variant_acc=res.variant_acc.round(3), t=res.t.round(2))
      .pivot(index="comparison", columns="model", values="p").round(3).to_string())
print()
print((res.pivot(index="comparison", columns="model", values="diff") * 100).round(2).to_string())
