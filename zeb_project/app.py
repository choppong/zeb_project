import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent  # 🔸 app.py가 있는 폴더 경로

# -----------------------------
# 데이터 불러오기 (캐싱)
# -----------------------------
@st.cache_data
def load_data():
    data_path = BASE_DIR / "data" / "zeb_clean.csv"   # 🔸 app.py 기준으로 찾기
    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding="cp949")
        return df
    else:
        return None


# -----------------------------
# 모델 불러오기 (캐싱)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "model_binary.pkl"   # 🔸 app.py 옆에 있는 파일
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print("모델 로드 오류:", e)
            return None
    else:
        return None



# -----------------------------
# 페이지 설정
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

# 탭 구성: 데이터 / 설명 / 예측
tab1, tab2, tab3 = st.tabs(
    ["📊 데이터 살펴보기", "🧪 연구/모델 설명", "🤖 예측하기 데모"]
)

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
          · 건물 용도, 지역, 연면적, 주거/비주거, 인증 구분 등 다양한 인자를 포함  

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

        추후에는 다음 기능을 추가/개선할 수 있습니다.
        - 더 많은 입력 항목(외피 성능, 설비 시스템 등)을 반영한 상세 예측
        - 다양한 시나리오별 에너지 절감 효과 및 등급 변화 분석
        """
    )

    st.info(
        "문의 / 피드백은 발표자에게 직접 말씀해 주세요 😊\n\n"
        "이 페이지는 Streamlit Community Cloud를 이용해 배포되었습니다."
    )


# ---------------------------------
# 탭 3: 예측하기 데모
# ---------------------------------
with tab3:
    st.subheader("제로에너지건축물 인증 등급 예측 데모")

    model = load_model()
    if model is None:
        st.error("`model_binary.pkl` 파일을 찾을 수 없거나, 로드 중 오류가 발생했습니다.")
    else:
        st.markdown(
            """
            아래 항목들을 입력하면 **학습된 모델을 이용해 예측 결과**를 보여주는 데모입니다.  
            실제 학습에 사용된 특성과 1:1로 완전히 일치하지 않을 수 있으며,  
            졸업작품 발표용 **컨셉 데모**임을 참고해 주세요 🙂
            """
        )

        col1, col2 = st.columns(2)

        # ------------------------------
        # 왼쪽: 건축물 기본 정보
        # ------------------------------
        with col1:
            bld_name = st.text_input("건축물명", value="예) 신세종빛드림본부 종합사무실")

            cert_type = st.selectbox("인증 구분", ["본인증", "예비인증"])

            res_type = st.selectbox("주거용 / 주거용 이외", ["주거용", "주거용 이외"])

            bld_use = st.selectbox(
                "건물 용도",
                [
                    "단독주택",
                    "공동주택",
                    "제1종 근린생활시설",
                    "제2종 근린생활시설",
                    "문화 및 집회시설",
                    "종교시설",
                    "판매시설",
                    "운수시설",
                    "의료시설",
                    "교육연구시설",
                    "노유자시설",
                    "수련시설",
                    "운동시설",
                    "업무시설",
                    "숙박시설",
                    "위락시설",
                    "공장",
                ],
            )

        # ------------------------------
        # 오른쪽: 지역 / 면적
        # ------------------------------
        with col2:
            region = st.selectbox(
                "지역",
                [
                    "서울",
                    "부산",
                    "대구",
                    "인천",
                    "광주",
                    "대전",
                    "울산",
                    "세종",
                    "경기",
                    "강원",
                    "충북",
                    "충남",
                    "전북",
                    "전남",
                    "경북",
                    "경남",
                    "제주",
                ],
            )

            area = st.number_input(
                "연면적 (m²)",
                min_value=50.0,
                max_value=300000.0,
                value=5000.0,
                step=50.0,
            )

            st.markdown("---")

        # ------------------------------
        # 예측 버튼
        # ------------------------------
        if st.button("✨ 예측하기"):

            # 1) 모델이 학습할 때 사용한 feature 이름 목록 가져오기
            if not hasattr(model, "feature_names_in_"):
                st.error("모델에 feature_names_in_ 정보가 없어 예측을 수행할 수 없습니다.")
            else:
                feature_names = list(model.feature_names_in_)
                # 먼저 모든 피처를 0으로 초기화
                data = {f: 0 for f in feature_names}

                # 2) 연속형 변수 채우기
                if "연면적" in data:
                    data["연면적"] = area

                # 3) 인증 구분 (본인증 / 예비인증)
                if cert_type == "본인증":
                    if "인증구분_본인증" in data:
                        data["인증구분_본인증"] = 1
                elif cert_type == "예비인증":
                    if "인증구분_예비인증" in data:
                        data["인증구분_예비인증"] = 1

                # 4) 주거용 / 주거용 이외
                if res_type == "주거용":
                    if "건물구분_주거용" in data:
                        data["건물구분_주거용"] = 1
                elif res_type == "주거용 이외":
                    if "건물구분_주거용 이외" in data:
                        data["건물구분_주거용 이외"] = 1

                # 5) 건물 용도 17개 → one-hot
                use_map = {
                    "단독주택": "건물용도_단독주택",
                    "공동주택": "건물용도_공동주택",
                    "제1종 근린생활시설": "건물용도_제1종 근린생활시설",
                    "제2종 근린생활시설": "건물용도_제2종 근린생활시설",
                    "문화 및 집회시설": "건물용도_문화 및 집회시설",
                    "종교시설": "건물용도_종교시설",
                    "판매시설": "건물용도_판매시설",
                    "운수시설": "건물용도_운수시설",
                    "의료시설": "건물용도_의료시설",
                    "교육연구시설": "건물용도_교육연구시설",
                    "노유자시설": "건물용도_노유자시설",
                    "수련시설": "건물용도_수련시설",
                    "운동시설": "건물용도_운동시설",
                    "업무시설": "건물용도_업무시설",
                    "숙박시설": "건물용도_숙박시설",
                    "위락시설": "건물용도_위락시설",
                    "공장": "건물용도_공장",
                }
                use_col = use_map.get(bld_use)
                if use_col and use_col in data:
                    data[use_col] = 1

                # 6) 지역 17개 → one-hot
                region_map = {
                    "서울": "지역_서울",
                    "부산": "지역_부산",
                    "대구": "지역_대구",
                    "인천": "지역_인천",
                    "광주": "지역_광주",
                    "대전": "지역_대전",
                    "울산": "지역_울산",
                    "세종": "지역_세종",
                    "경기": "지역_경기",
                    "강원": "지역_강원",
                    "충북": "지역_충북",
                    "충남": "지역_충남",
                    "전북": "지역_전북",
                    "전남": "지역_전남",
                    "경북": "지역_경북",
                    "경남": "지역_경남",
                    "제주": "지역_제주",
                }
                region_col = region_map.get(region)
                if region_col and region_col in data:
                    data[region_col] = 1

                   # 7) DataFrame으로 변환 (컬럼 순서까지 model에 맞게)
    input_df = pd.DataFrame([data])[feature_names]

    # 8) 예측 수행 (심플 버전 – 에러 처리 없이 바로 예측)
    pred = model.predict(input_df)[0]

    # 확률값(있으면) 계산
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]

    # 9) 0/1 예측값을 사람이 이해하기 쉬운 등급 표현으로 변환
    try:
        pred_int = int(pred)
    except Exception:
        pred_int = pred

    if pred_int == 1:
        grade_text = "1~3등급 (예상 상위 등급)"
    elif pred_int == 0:
        grade_text = "4·5등급 (예상 하위 등급)"
    else:
        grade_text = f"모델 원시 예측값: {pred}"

    # 10) 화면에 출력
    st.success(f"이 건축물의 예측 결과: **{grade_text}**")

    if proba is not None:
        st.write(
            f"해당 결과(1~3등급일 확률, 양성 클래스 기준): **{proba:.1%}**"
        )

    st.caption(
        "※ 예측 결과는 연구/발표용 참고값입니다. 실제 인증과는 차이가 있을 수 있습니다."
    )


       






