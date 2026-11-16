"""결측 처리 4(Mean/Median/Most-Frequent/Drop NA) x 인코딩 2 x 스케일링 4 x 변수 선택 2 x 파생변수 2 = 128개 조합 완전 교차 실험.
step2_experiments.py 의 run() 재사용."""
import itertools
import pandas as pd
from joblib import Parallel, delayed
from common import MODELS, OUT
from step2_experiments import run

jobs = []
for imp, enc, sc, sel, fe in itertools.product(["mean", "median", "most_frequent", "dropna"], ["onehot", "label"],
                                               ["standard", "minmax", "robust", "none"], [False, True], [False, True]):
    for m in MODELS:
        cfg = dict(impute="median" if imp == "dropna" else imp, encode=enc, scale=sc, select=sel, use_fe=fe,
                   drop=(imp == "dropna"))
        jobs.append((imp, m, cfg))
res = Parallel(n_jobs=-1, verbose=3)(delayed(run)(i, m, c) for i, m, c in jobs)
out = pd.DataFrame(res).rename(columns={"exp": "imputer"})
out.to_csv(f"{OUT}/dropna_grid.csv", index=False)
print(len(out), "rows saved")
