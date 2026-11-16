# Titanic Feature Engineering Pipeline

- 데이터: Kaggle Titanic train.csv (891행) -> `data/titanic.csv`
- 최종 제출 보고서: `report.pdf` (코드가 만든 초안을 직접 다듬은 최종본)
- 실행: `pip install -r requirements.txt` 후 `python run_all.py` (SEED=42 고정, 약 20분)
  - 코드가 만드는 보고서 초안은 `report_generated.pdf` (최종본 `report.pdf`를 덮어쓰지 않음)
  - AutoML(FLAML)은 시간 예산 기반 탐색이라 재실행하면 수치가 조금 달라질 수 있음
- 한글 경로 환경이면 `JOBLIB_TEMP_FOLDER`를 영문 경로로 지정
- 단계별 파일: `step1_eda.py`(EDA) -> `step2_experiments.py`, `step2b_dropna_grid.py`(실험) -> `step3_select_tune_shap.py`(변수 선택·GridSearchCV·SHAP) -> `step5_automl.py`(AutoML) -> `step6_stat_tests.py`(통계 검정) -> `step4_report.py`(보고서 초안)
- 결과: `outputs/` (그림, CSV, JSON)
