"""공통 모듈: 데이터 로드, 파생 변수, 파이프라인 빌더 (Titanic)."""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler, StandardScaler)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
SEED = 42
DATA = "data/titanic.csv"
OUT = "outputs"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def load_data():
    return pd.read_csv(DATA)


# ---------------------------------------------------------------- 파생 변수
RARE = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """파생 변수 생성(무상태). use_fe=False 이면 원본 변수만 사용한다."""

    def __init__(self, use_fe=True):
        self.use_fe = use_fe

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        out["Pclass"] = X["Pclass"]
        out["Sex"] = X["Sex"]
        out["Age"] = X["Age"]
        out["SibSp"] = X["SibSp"]
        out["Parch"] = X["Parch"]
        out["Fare"] = X["Fare"]
        out["Embarked"] = X["Embarked"]
        if self.use_fe:
            fam = X["SibSp"] + X["Parch"] + 1
            out["FamilySize"] = fam                                  # 파생1
            out["IsAlone"] = (fam == 1).astype(int)                  # 파생2
            out["FarePerPerson"] = X["Fare"] / fam                   # 파생3
            out["HasCabin"] = X["Cabin"].notna().astype(int)         # 파생4
            title = X["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
            title = title.replace(RARE)
            common = ["Mr", "Mrs", "Miss", "Master"]
            out["Title"] = title.where(title.isin(common), "Rare")   # 파생5
            out["AgeGroup"] = pd.cut(                                # 파생6
                X["Age"], [0, 12, 18, 35, 60, 120],
                labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"]
            ).astype(object).fillna("Unknown")
        return out


NUM_RAW = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CAT_RAW = ["Sex", "Embarked"]
NUM_FE = ["FamilySize", "IsAlone", "FarePerPerson", "HasCabin"]
CAT_FE = ["Title", "AgeGroup"]
BASE_COLS = ["Pclass", "SibSp", "Parch", "Fare"]   # 결측 없는 수치형만 (Base)

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                     random_state=SEED, n_jobs=1, verbosity=0),
    "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                       random_state=SEED, n_jobs=1, verbose=-1),
}
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler, "none": None}


def build_pipeline(model_name, impute="median", encode="onehot", scale="standard",
                   select=False, use_fe=True, top_n=8, model=None, base=False):
    """전체 ML Pipeline 객체를 생성한다.

    impute : mean | median | most_frequent  (수치형 결측치 전략; 범주형은 항상 most_frequent)
    encode : onehot | label
    scale  : standard | minmax | robust | none  (수치형에만 적용)
    select : True 이면 RF 중요도 상위 top_n개 변수만 사용 (SelectFromModel, 파이프라인 내부 → 누수 없음)
    base   : True 이면 결측 없는 원본 수치형 4개만 사용(전처리 없음)
    """
    est = model if model is not None else MODELS[model_name]()
    if base:
        cols = BASE_COLS
        return Pipeline([("sel_cols", ColumnTransformer([("keep", "passthrough", cols)])),
                         ("model", est)])

    num = NUM_RAW + (NUM_FE if use_fe else [])
    cat = CAT_RAW + (CAT_FE if use_fe else [])
    num_steps = [("imp", SimpleImputer(strategy=impute))]
    if SCALERS[scale] is not None:
        num_steps.append(("scale", SCALERS[scale]()))
    if encode == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_steps = [("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]
    pre = ColumnTransformer([("num", Pipeline(num_steps), num),
                             ("cat", Pipeline(cat_steps), cat)])
    steps = [("fe", FeatureEngineer(use_fe)), ("pre", pre)]
    if select:
        steps.append(("select", SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
            threshold=-np.inf, max_features=top_n)))
    steps.append(("model", est))
    return Pipeline(steps)


def feature_names(pipe):
    """ColumnTransformer 이후 변수명 (선택 단계 반영)."""
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    names = [n.split("__", 1)[1] for n in names]
    if "select" in pipe.named_steps:
        names = list(np.array(names)[pipe.named_steps["select"].get_support()])
    return names
