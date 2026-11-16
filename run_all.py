"""전체 재현: python run_all.py  (어느 폴더에서 실행해도 됨, 데이터는 data/titanic.csv 필요)"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["step1_eda.py", "step2_experiments.py", "step2b_dropna_grid.py", "step3_select_tune_shap.py",
         "step4_automl.py", "step5_stat_tests.py"]
for s in STEPS:
    print(f"=== {s} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / s)], check=True, cwd=HERE)
