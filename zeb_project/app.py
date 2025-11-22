import streamlit as st
import pandas as pd
from pathlib import Path
import joblib  # 🔹 추가


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
        @st.cache_resource
def load_model():
    model_path = Path("model_binary.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
        return model
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

tab1, tab2, tab3 = st.tabs(["📊 데이터 살펴보기", "🧪 연구/모델 설명", "🤖 예측하기 데모"])


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


with tab3:
    st.subheader("제로에너지건축물 인증 등급 예측 데모")

    model = load_model()
    if model is None:
        st.error("`model_binary.pkl` 파일을 찾을 수 없습니다. GitHub 레포에 모델 파일이 있는지 확인해 주세요.")
    else:
        st.markdown(
            """
            아래 항목들을 입력하면 **학습된 모델을 이용해 예측 결과**를 보여주는 데모입니다.  
            실제 학습에 사용된 특성과 1:1로 완전히 일치하지 않을 수 있으며,  
            졸업작품 발표용 **컨셉 데모**임을 참고해 주세요 🙂
            """
        )

        col1, col2 = st.columns(2)

        # 🔹 왼쪽: 건축물 기본 정보
        with col1:
            bld_name = st.text_input("건축물명", value="예) 신세종빛드림본부 종합사무실")
            cert_type = st.selectbox("인증 구분", ["본인증", "예비인증"])
            res_type = st.selectbox("주거용 / 주거용 이외", ["주거용", "주거용 이외"])

        # 🔹 오른쪽: 용도/지역/면적
        with col2:
            bld_use = st.selectbox(
                "건물 용도",
                [
                    "업무시설",
                    "교육연구시설",
                    "제1종근린생활시설",
                    "제2종근린생활시설",
                    "문화및집회시설",
                    "판매시설",
                    "숙박시설",
                    "위락시설",
                    "공장",
                    "창고시설",
                    "운수시설",
                    "의료시설",
                    "노유자시설",
                    "수련시설",
                    "위험물저장및처리시설",
                    "자동차관련시설",
                    "기타"
                ]  # ← 실제 17개 목록으로 수정해도 돼
            )

            region = st.selectbox("지역", ["수도권", "충청권", "호남권", "영남권", "강원·제주", "기타"])
            area = st.number_input("연면적 (m²)", min_value=50.0, max_value=300000.0, value=5000.0, step=50.0)

        st.markdown("---")

        if st.button("✨ 예측하기"):
            # 🔻 실제 학습에 사용했던 컬럼 이름/전처리 방식에 맞게 수정해야 하는 부분!
            input_dict = {
                # 예시) 인증 구분 더미 변수
                "인증_본인증": 1 if cert_type == "본인증" else 0,
                "인증_예비인증": 1 if cert_type == "예비인증" else 0,

                # 예시) 주거/비주거 더미
                "주거용여부": 1 if res_type == "주거용" else 0,

                # 예시) 용도 더미 (실제 학습 컬럼명에 맞게 바꿔야 함)
                "용도_업무시설": 1 if bld_use == "업무시설" else 0,
                "용도_교육연구시설": 1 if bld_use == "교육연구시설" else 0,
                "용도_제1종근린생활시설": 1 if bld_use == "제1종근린생활시설" else 0,
                # ... 나머지 용도도 같은 패턴으로 추가

                # 예시) 지역 더미
                "지역_수도권": 1 if region == "수도권" else 0,
                "지역_충청권": 1 if region == "충청권" else 0,
                "지역_호남권": 1 if region == "호남권" else 0,
                "지역_영남권": 1 if region == "영남권" else 0,
                "지역_강원제주": 1 if region == "강원·제주" else 0,

                # 연속형 변수
                "연면적": area,
            }

            input_df = pd.DataFrame([input_dict])

            try:
                pred = model.predict(input_df)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0][1]
                else:
                    proba = None
            except Exception as e:
                st.error("모델 예측 중 오류가 발생했습니다. 피처 이름/개수, 더미 변수 구성을 확인해 주세요.")
                st.exception(e)
            else:
                st.success(f"이 건축물의 예측 결과: **{pred}**")

                if proba is not None:
                    st.write(f"해당 결과일 확률(양성 클래스 기준): **{proba:.1%}**")

                st.caption("※ 예측 결과는 연구/발표용 참고값입니다. 실제 인증과는 차이가 있을 수 있습니다.")
