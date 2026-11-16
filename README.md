# Titanic Feature Engineering Pipeline

Titanic 생존 예측(분류)에서 결측치 처리·인코딩·스케일링·파생변수·변수 선택 조합을 비교하는 실험 코드입니다.

## 실행 방법

가상환경은 필수는 아니지만, 이미 설치된 다른 패키지와 버전이 충돌하지 않도록 **권장**합니다.

```bash
# 1) (권장) 가상환경 만들기
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 전체 실행 (약 20분, 어느 폴더에서 실행해도 됨)
python run_all.py
```

단계별로 따로 실행할 수도 있습니다. 순서는 아래 번호를 지켜 주세요(앞 단계의 결과 파일을 뒤 단계가 읽습니다).

| 순서 | 파일 | 내용 | 결과 |
|---|---|---|---|
| 1 | `step1_eda.py` | EDA와 시각화 | `outputs/eda_*.png`, `eda_summary.json` |
| 2 | `step2_experiments.py` | Base/Exp 실험, 96개 조합 그리드 | `outputs/all_results.csv` |
| 2b | `step2b_dropna_grid.py` | 결측 처리 4가지(Drop NA 포함) 교차 실험 | `outputs/dropna_grid.csv` |
| 3 | `step3_select_tune_shap.py` | 변수 선택, GridSearchCV, SHAP | `outputs/fs_*`, `shap_*`, `step3_summary.json` |
| 4 | `step4_automl.py` | AutoML(FLAML) 비교 | `outputs/automl_results.csv` |
| 5 | `step5_stat_tests.py` | 요인별 차이의 통계 검정 | `outputs/stat_tests.csv` |

공통 코드는 `common.py`(데이터 로드, 파생변수, 파이프라인 구성)입니다.

## 확인된 환경과 주의할 점

- 실행해 확인한 환경: Windows 11, Python 3.14.3, pandas 3.0.5, numpy 2.4.4, scikit-learn 1.9.0, xgboost 3.4.1, lightgbm 4.7.0, shap 0.54.1, flaml 2.8.0
- 재현성: `random_state=42`를 고정했습니다. 같은 환경이면 결과가 같습니다. 단, **AutoML(`step4_automl.py`)은 시간 예산(60초) 기반 탐색이라 실행할 때마다 수치가 조금씩 달라질 수 있습니다.** 라이브러리 버전이 다르면 소수점 셋째 자리 정도가 달라질 수 있습니다.
- 그림의 한글 글꼴: Windows(맑은 고딕), macOS(AppleGothic), Linux(`fonts-nanum` 등)를 자동으로 찾습니다. 글꼴이 없으면 경고가 나오고 그림의 한글이 깨질 수 있습니다.
- 사용자 폴더 이름에 한글이 있으면 병렬 실행(joblib)이 실패할 수 있어, 임시 폴더를 영문 경로(`joblib_tmp`)로 자동 전환합니다.
- macOS에서 LightGBM이 `libomp` 오류를 내면 `brew install libomp` 후 다시 실행하세요.
- 데이터: Kaggle Titanic `train.csv`(891행)가 `data/titanic.csv`로 들어 있습니다.
