import pandas as pd

# CSV 파일 경로
DATA_PATH = "data/zeb_data.csv"

def load_data(path=DATA_PATH):
    # 한글 CSV 인코딩 문제 방지용
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    return df


if __name__ == "__main__":
    df = load_data()

    print("===== 데이터 첫 5줄 =====")
    print(df.head())

    print("\n===== 데이터 정보 =====")
    print(df.info())

    print("\n===== 컬럼 이름 =====")
    print(df.columns.tolist())

    # ===== 필요한 컬럼만 선택 =====
    selected_columns = ['인증구분', '건물구분', '건물용도', '지역', '연면적', '인증등급']
    df = df[selected_columns]

    # ===== 결측치 제거 =====
    df = df.dropna()

    print("\n===== 결측치 제거 후 =====")
    print(df.info())

    # ===== 데이터 타입 변환 (연면적, 인증등급 숫자로) =====
    df['연면적'] = pd.to_numeric(df['연면적'], errors='coerce')
    df['인증등급'] = pd.to_numeric(df['인증등급'], errors='coerce', downcast='integer')

    # 숫자로 못 바뀐 값이 있으면 NaN → 제거
    df = df.dropna()

    print("\n===== 타입/결측치 정리 후 =====")
    print(df.info())

    print("\n===== 필요한 컬럼만 남긴 후 =====")
    print(df.head())
    print(df.info())
    # ===== 전처리 완료 데이터 저장 =====
    df.to_csv("data/zeb_clean.csv", index=False, encoding="utf-8-sig")


