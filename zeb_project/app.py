import streamlit as st
import pandas as pd
from pathlib import Path

# -----------------------------
# 데이터 불러오기 (캐싱)
# -----------------------------
@st.cache_data
def load_data():
    data_path = Path("data") / "zeb_clean.csv"
    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
        except UnicodeDecodeError:
            # 인코딩 문제 있을 경우 대비
            df = pd.read_csv(data_path, encoding="cp949")
        return df
    else:
        return None

# -----------------------------
# 앱 화면 구성
# -----------------------------
st.set_page_config(
    page_title="ZEB 인증등급 예측 졸업작품",
    layout="wide"
)

st.title("제로에너지건축물 인증등급 예측 프로그램")
st.caption("한양대학교 ERICA · 졸업작품 데모 페이지")

st.markdown(
    """
    이 웹페이지는 졸업작품 **“제로에너지건축물인증 등급 예측을 위한 
    데이터 기반 프로그램 개발 및 적용에 관한 연구”**의 데모 화면입니다.  

    판넬에 있는 QR 코드를 통해 접속하신 분들은 아래 내용을 참고해 주세요 😊
    """
)

tab1, tab2 = st.tabs(["📊 데이터 살펴보기", "🧪 연구/모델 설명"])

# ---------------------------------
# 탭 1: 데이터 살펴보기
# ---------------------------------
with tab1:
    st.subheader("사용된 제로에너지건축물 데이터")

    df = load_data()
    if df is None:
        st.warning("`data/zeb_clean.csv` 파일을 찾을 수 없습니다. (GitHub 레포의 data 폴더를 확인해 주세요.)")
    else:
        st.write(f"총 **{len(df):,}건**의 데이터가 있습니다.")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("### 컬럼(특성) 정보")
        st.write(list(df.columns))

        with st.expander("전체 데이터 간단 통계 보기"):
            st.write(df.describe(include="all"))

# ---------------------------------
# 탭 2: 연구/모델 설명
# ---------------------------------
with tab2:
    st.subheader("연구 개요")

    st.markdown(
        """
        - **연구 목적**  
          · 건물 에너지 성능 및 관련 인자를 기반으로 제로에너지건축물 인증 등급을 예측하는  
            데이터 기반 프로그램을 개발하였습니다.  

        - **데이터 구성**  
          · 제로에너지건축물 인증 대상 건물들의 설계/운영 정보를 정리한 데이터셋 활용  
          · 에너지성능지표, 설비특성, 외피성능 등 여러 인자를 포함  

        - **모델 개요**  
          · Python 기반 머신러닝 기법 활용  
          · 학습된 모델 파일(`model_binary.pkl`)을 통해 인증 등급/합격 여부 예측  
        """
    )

    st.markdown("---")

    st.markdown(
        """
        이 페이지는 데모 버전으로,  
        **졸업작품 발표 및 판넬 관람객에게 연구 내용을 소개하는 용도**로 제작되었습니다.  

        추후에는 다음 기능을 추가할 수 있습니다.
        - 사용자가 직접 건물 조건(지역, 용도, 외피성능 등)을 입력하고  
          → 예상 ZEB 인증 등급을 확인하는 인터랙티브 예측 기능
        - 다양한 시나리오별 에너지 절감 효과 비교
        """
    )

    st.info(
        "문의 / 피드백은 발표자에게 직접 말씀해 주세요 😊\n\n"
        "이 페이지는 Streamlit Community Cloud를 이용해 배포되었습니다."
    )

