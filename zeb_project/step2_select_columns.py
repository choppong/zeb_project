import pandas as pd

# 1) CSV 경로
DATA_PATH = "data/zeb_data.csv"

def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    return df

if __name__ == "__main__":
    df = load_data()

    # 필요한 5개 컬럼만 선택
    cols = ["건물용도", "지역", "연면적", "인증등급", "건물명"]
    df_selected = df[cols].copy()

    print("===== 선택한 5개 컬럼 5줄 =====")
    print(df_selected.head())

    print("\n===== 각 컬럼 데이터 타입 =====")
    print(df_selected.dtypes)

    # X, y 분리
    X = df_selected[["건물용도", "지역", "연면적"]]
    y = df_selected["인증등급"]

    print("\n===== X 첫 5줄 =====")
    print(X.head())

    print("\n===== y 첫 5줄 =====")
    print(y.head())
