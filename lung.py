import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ====================================================================
# [파일 로드] lung.py와 같은 폴더에 파일들이 있다고 가정합니다.
# ====================================================================
@st.cache_resource
def load_saved_files():
    scaler = joblib.load('lung_scaler.pkl')      # 스케일러 파일 불러오기
    model = joblib.load('lung_model.pkl')        # 학습된 모델 파일 불러오기
    df = pd.read_csv('lung.csv')       # 원본 데이터(CSV) 불러오기
    return scaler, model, df

try:
    scaler, model, df = load_saved_files()
except FileNotFoundError:
    st.error("🚨 `scaler.pkl`, `model.pkl`, `lung_data.csv` 파일을 찾을 수 없습니다!")
    st.stop()
# ====================================================================

st.title("환자 군집 예측 및 시각화 프로그램")
st.write("환자의 정보를 입력하면 어떤 군집에 속하는지 예측하고 그래프에 표시합니다.")

st.divider()

## 1. 데이터 입력 (사이드바)
st.sidebar.header("환자 정보 입력")

Age = st.sidebar.number_input("나이 입력 (Age)", min_value=0, max_value=120, value=30, step=1)
Smokes = st.sidebar.number_input("흡연량 입력 (Smokes)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
AreaQ = st.sidebar.number_input("지역지수 입력 (AreaQ)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
Alkhol = st.sidebar.number_input("음주량 입력 (Alkhol)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# 분석 시작 버튼
if st.sidebar.button("군집 예측하기"):
    
    ## 2. 데이터프레임 변환 및 [수정] 넘파이 배열로 스케일링 진행
    # 스케일러가 기억하는 원래 컬럼의 개수를 파악합니다.
    try:
        expected_features = scaler.n_features_in_
    except AttributeError:
        expected_features = 4  # 기본값으로 4개 가정 (Age, Smokes, AreaQ, Alkhol)

    # 1) 만약 원래 학습 때 컬럼이 4개였다면
    if expected_features == 4:
        new_patient = pd.DataFrame([[Age, Smokes, AreaQ, Alkhol]])
    # 2) 만약 원래 데이터셋의 전체 컬럼(7개)을 다 넣고 학습시켰었다면 (Name, Surname 포함)
    elif expected_features == 7:
        # 문자열 자리에 임의의 값(0)을 채워서 개수를 7개로 맞춰줍니다.
        new_patient = pd.DataFrame([[0, 0, Age, Smokes, AreaQ, Alkhol, 0]])
    # 3) 만약 스케일러가 기억하는 개수가 3개였다면 (Smokes, AreaQ, Alkhol)
    else:
        new_patient = pd.DataFrame([[Smokes, AreaQ, Alkhol]])
    
    # 🌟 핵심 수정: .values를 붙여 컬럼 이름 매칭 검사를 강제로 건너뜁니다!
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)
    
    ## 3. 결과 출력
    st.subheader("🔮 예측 결과")
    st.success(f"이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")
    
    st.divider()
    
    ## 4. 산점도 시각화
    st.subheader("📊 군집 분포도 내 환자 위치")
    
    # 한글 깨짐 방지
    plt.rc('font', family='Malgun Gothic') 
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 데이터셋의 군집 번호가 'Result' 컬럼에 있으므로 이를 색상(c)으로 지정합니다.
    cluster_column = 'Result' if 'Result' in df.columns else 'cluster'
    
    scatter = ax.scatter(df['Smokes'], df['Alkhol'], c=df[cluster_column], alpha=0.5, cmap='viridis')
    
    # 새 환자 표시 (X 기호)
    ax.scatter(Smokes, Alkhol, c='red', s=300, marker='X', label='New Patient')
    
    ax.set_xlabel('흡연량 (Smokes)')
    ax.set_ylabel('음주량 (Alkhol)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 스트림릿에 plot 표시
    st.pyplot(fig)
