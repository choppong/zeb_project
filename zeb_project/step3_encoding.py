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


def make_X_y(df):
    """문자형 컬럼 인코딩하고 X(입력), y(정답) 나누기"""
    # 범주형(문자) 컬럼 목록
    categorical_cols = ["인증구분", "건물구분", "건물용도", "지역"]

    # 원-핫 인코딩 (문자 → 0/1 컬럼 여러 개로)
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # 입력(X), 타깃(y) 분리
    X = df_encoded.drop("인증등급", axis=1)
    y = df_encoded["인증등급"]

    return X, y

def train_gbm(X, y, test_size=0.2, random_state=42):
    """Gradient Boosting 모델 학습 + 평가 (희귀 등급에 가중치 부여)"""
    # 1) 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # 각 등급 비율 유지
    )

    # 2) 학습 데이터에서 등급별 개수 확인
    class_counts = y_train.value_counts().sort_index()
    print("\n===== 학습 데이터 등급 분포 =====")
    print(class_counts)

    # 3) 등급별 가중치 계산 (개수 적을수록 가중치 ↑)
    #    예: weight_k = 전체샘플 / (클래스수 * 해당등급개수)
    n_classes = len(class_counts)
    total = len(y_train)
    class_weight = total / (n_classes * class_counts)
    print("\n===== 등급별 가중치 =====")
    print(class_weight)

    # 4) 각 샘플별 weight 만들기
    sample_weight = y_train.map(class_weight)

    # 5) 모델 학습 (가중치 적용)
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # 6) 예측 및 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n===== 테스트 정확도 =====")
    print(f"{acc:.3f}")

    print("\n===== 등급별 성능(분류 리포트) =====")
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    # 1) 데이터 불러오기
    df = load_clean_data()
    print("===== 전처리된 데이터 첫 5줄 =====")
    print(df.head())

    # 2) 인코딩 + X, y 만들기
    X, y = make_X_y(df)
    print("\nX shape:", X.shape)
    print("y value counts:")
    print(y.value_counts().sort_index())

    # 3) GBM 모델 학습 및 평가
    model = train_gbm(X, y)
