"""결측 처리 4(Mean/Median/Most-Frequent/Drop NA) x 인코딩 2 x 스케일링 4 x 변수 선택 2 x 파생변수 2 = 128개 조합 완전 교차 실험.
Mean/Median/Most-Frequent 384개 결과는 step2_experiments.py 가 이미 계산한 all_results.csv(그리드)를 재사용하고,
평가 데이터가 달라 따로 계산해야 하는 Drop NA 128개만 새로 학습한다. (step2_experiments.py 를 먼저 실행해야 함)"""
import itertools
import pandas as pd
from joblib import Parallel, delayed
from common import MODELS, OUT
from step2_experiments import run

jobs = []
for enc, sc, sel, fe in itertools.product(["onehot", "label"], ["standard", "minmax", "robust", "none"], [False, True], [False, True]):
    for m in MODELS:
        jobs.append(("dropna", m, dict(impute="median", encode=enc, scale=sc, select=sel, use_fe=fe, drop=True)))
new = pd.DataFrame(Parallel(n_jobs=-1, verbose=3)(delayed(run)(i, m, c) for i, m, c in jobs)).rename(columns={"exp": "imputer"})

prev = pd.read_csv(f"{OUT}/all_results.csv")
prev = prev[prev.group == "B"].copy()
prev["imputer"] = prev["impute"]
prev["drop"] = False
out = pd.concat([prev[new.columns], new], ignore_index=True)
order = {"mean": 0, "median": 1, "most_frequent": 2, "dropna": 3}
out = out.sort_values("imputer", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
out.to_csv(f"{OUT}/dropna_grid.csv", index=False)
print(len(out), "rows saved (reused", len(prev), "+ new", len(new), ")")
