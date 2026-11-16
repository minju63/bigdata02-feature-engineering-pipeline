"""전체 재현: python run_all.py  (데이터는 data/titanic.csv 필요)"""
import subprocess, sys
for s in ["step1_eda.py", "step2_experiments.py", "step2b_dropna_grid.py", "step3_select_tune_shap.py", "step4_automl.py", "step5_stat_tests.py"]:
    subprocess.run([sys.executable, s], check=True)
