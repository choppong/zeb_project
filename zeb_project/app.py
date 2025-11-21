import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from flask import Flask, render_template_string, request

# ===== 전처리된 데이터 경로 =====
DATA_PATH = "data/zeb_clean.csv"

# 전역에서 사용할 변수들
CATEGORICAL_COLS = ["인증구분", "건물구분", "건물용도", "지역"]


def train_binary_model(path=DATA_PATH):
    """전처리 완료된 CSV 불러와서 이진 분류 모델 학습"""
    df = pd.read_csv(path)

    # 상위등급(1~3) = 1, 일반등급(4~5) = 0
    df["상위등급"] = (df["인증등급"] <= 3).astype(int)

    # 범주형 칼럼 옵션(드롭다운에 쓰기 위함)
    options = {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_COLS}

    # 원-핫 인코딩
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    X = df_encoded.drop(["인증등급", "상위등급"], axis=1)
    y = df_encoded["상위등급"]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    feature_columns = X.columns.tolist()

    return model, feature_columns, options


# 앱 시작할 때 한 번만 학습
model, FEATURE_COLUMNS, OPTIONS = train_binary_model()

app = Flask(__name__)

# 아주 간단한 HTML 템플릿 (Flask 안에서 문자열로 바로 사용)
TEMPLATE = """
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <title>ZEB 예상 등급 예측</title>
    <style>
      body { font-family: sans-serif; background:#f5f7fb; }
      .container { max-width: 700px; margin:40px auto; padding:24px; background:white; border-radius:16px; box-shadow:0 4px 12px rgba(0,0,0,0.08); }
      h1 { font-size: 26px; margin-bottom: 16px; }
      label { font-weight: 600; }
      .row { display:flex; gap:16px; margin-bottom:12px; }
      .col { flex:1; display:flex; flex-direction:column; }
      select, input { padding:8px; border-radius:8px; border:1px solid #ccc; }
      button { margin-top:16px; padding:10px 18px; border-radius:999px; border:none; background:#ffcc33; font-weight:700; cursor:pointer; }
      .result { margin-top:24px; padding:16px; border-radius:12px; }
      .high { background:#e7f5ff; border:1px solid #4dabf7; }
      .normal { background:#fff9db; border:1px solid #fcc419; }
      .prob { font-size:14px; color:#555; margin-top:4px; }
      .note { margin-top: 10px; font-size: 12px; color: #888;}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>ZEB 예상 등급(상위 vs 일반) 예측</h1>
      <form method="post">
        <div class="row">
          <div class="col">
            <label>인증구분</label>
            <select name="인증구분">
              {% for v in options["인증구분"] %}
              <option value="{{v}}">{{v}}</option>
              {% endfor %}
            </select>
          </div>
          <div class="col">
            <label>건물구분</label>
            <select name="건물구분">
              {% for v in options["건물구분"] %}
              <option value="{{v}}">{{v}}</option>
              {% endfor %}
            </select>
          </div>
        </div>

        <div class="row">
          <div class="col">
            <label>건물용도</label>
            <select name="건물용도">
              {% for v in options["건물용도"] %}
              <option value="{{v}}">{{v}}</option>
              {% endfor %}
            </select>
          </div>
          <div class="col">
            <label>지역</label>
            <select name="지역">
              {% for v in options["지역"] %}
              <option value="{{v}}">{{v}}</option>
              {% endfor %}
            </select>
          </div>
        </div>

        <div class="row">
          <div class="col">
            <label>연면적 (㎡)</label>
            <input type="number" step="1" min="1" name="연면적" value="1000">
          </div>
        </div>

        <button type="submit">ZEB 예상 등급 예측하기</button>
      </form>

      {% if prediction is not none %}
      <div class="result {% if prediction == 1 %}high{% else %}normal{% endif %}">
        {% if prediction == 1 %}
          <strong>상위등급(1~3)으로 예측됩니다.</strong>
        {% else %}
          <strong>일반등급(4~5)으로 예측됩니다.</strong>
        {% endif %}
        {% if prob is not none %}
        <div class="prob">예측 확률: {{ (prob*100)|round(1) }}%</div>
        {% endif %}
      </div>
      <div class="note">※ 실제 인증 결과와는 차이가 있을 수 있으며, 데이터 기반 참고용입니다.</div>
      {% endif %}
    </div>
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None

    if request.method == "POST":
        # 폼에서 값 꺼내기
        input_data = {
            "인증구분": request.form.get("인증구분"),
            "건물구분": request.form.get("건물구분"),
            "건물용도": request.form.get("건물용도"),
            "지역": request.form.get("지역"),
            "연면적": float(request.form.get("연면적") or 0),
            "인증등급": 0,  # 자리 맞추기 용
        }

        # 1행짜리 데이터프레임으로 만들기
        input_df = pd.DataFrame([input_data])

        # 학습 때와 똑같이 인코딩
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS)
        input_encoded = input_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # 예측
        pred = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0][int(pred)]

        prediction = int(pred)
        prob = float(proba)

    return render_template_string(TEMPLATE, options=OPTIONS, prediction=prediction, prob=prob)


if __name__ == "__main__":
    # 로컬에서 테스트할 때만 사용 (Render에서는 gunicorn으로 실행)
    app.run(host="0.0.0.0", port=5000, debug=True)
