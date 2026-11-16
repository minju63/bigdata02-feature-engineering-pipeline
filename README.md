# Titanic Feature Engineering Pipeline

- 데이터: Kaggle Titanic train.csv (891행) -> `data/titanic.csv`
- 실행: `pip install -r requirements.txt` 후 `python run_all.py` (SEED=42 고정, 약 20분)
  - AutoML(FLAML)은 시간 예산 기반 탐색이라 재실행하면 수치가 조금 달라질 수 있음
- 한글 경로 환경이면 `JOBLIB_TEMP_FOLDER`를 영문 경로로 지정
- 단계별 파일: `step1_eda.py`(EDA) -> `step2_experiments.py`, `step2b_dropna_grid.py`(실험) -> `step3_select_tune_shap.py`(변수 선택·GridSearchCV·SHAP) -> `step5_automl.py`(AutoML) -> `step6_stat_tests.py`(통계 검정)
- 공통 코드: `common.py` (데이터 로드, 파생변수, 파이프라인 구성)
- 결과: `outputs/` (그림, CSV, JSON)
