import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===== 전처리된 데이터 경로 =====
DATA_PATH = "data/zeb_clean.csv"


def load_clean_data(path=DATA_PATH):
    """전처리 완료된 CSV 불러오기"""
    df = pd.read_csv(path)
    return df


def make_X_y_binary(df):
    """
    인증등급(1~5)을 이진 분류용 타깃으로 변환
    1,2,3 등급 -> 1 (상위등급)
    4,5 등급  -> 0 (일반등급)
    """

    # 이진 타깃 생성
    df = df.copy()
    df["상위등급"] = (df["인증등급"] <= 3).astype(int)

    # 범주형(문자) 컬럼 목록
    categorical_cols = ["인증구분", "건물구분", "건물용도", "지역"]

    # 원-핫 인코딩 (문자 → 0/1 더미 변수)
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # 입력(X), 타깃(y) 분리
    X = df_encoded.drop(["인증등급", "상위등급"], axis=1)
    y = df_encoded["상위등급"]

    return X, y


def train_binary_gbm(X, y, test_size=0.2, random_state=42):
    """상위등급(1) / 일반등급(0) 이진 분류 모델 학습 + 평가"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n===== 테스트 정확도 (이진 분류: 상위 vs 일반) =====")
    print(f"{acc:.3f}")

    print("\n===== 상위등급(1) / 일반등급(0) 분류 리포트 =====")
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    # 1) 데이터 불러오기
    df = load_clean_data()
    print("===== 전처리된 데이터 첫 5줄 =====")
    print(df.head())

    # 2) 이진 타깃 + 인코딩 + X, y 만들기
    X, y = make_X_y_binary(df)
    print("\nX shape:", X.shape)
    print("y value counts (0=일반, 1=상위):")
    print(y.value_counts().sort_index())

    # 3) GBM 이진 분류 모델 학습 및 평가
    model = train_binary_gbm(X, y)
