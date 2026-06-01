import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib  # 👈 파일을 불러오기 위해 꼭 필요합니다!

# ====================================================================
# ⚠️ [필수 수정] 내 컴퓨터에 있는 파일 이름(또는 전체 경로)으로 바꿔주세요!
# 만약 lung.py와 같은 폴더에 파일들이 저장되어 있다면 파일 이름만 적으셔도 됩니다.
# ====================================================================
@st.cache_resource  # 모델을 한 번만 불러오도록 속도를 최적화합니다.
def load_saved_files():
    # 예: 'C:/streamlit/scaler.pkl' 처럼 전체 경로를 적으셔도 됩니다.
    scaler = joblib.load('scaler.pkl')      # 스케일러 파일 불러오기
    model = joblib.load('model.pkl')        # 학습된 모델 파일 불러오기
    df = pd.read_csv('lung_data.csv')       # 원본 데이터(CSV) 불러오기
    return scaler, model, df

# 위 함수를 실행하여 파일들을 실제로 변수에 할당합니다.
try:
    scaler, model, df = load_saved_files()
except FileNotFoundError:
    st.error("🚨 `scaler.pkl`, `model.pkl`, `lung_data.csv` 파일을 찾을 수 없습니다! 파일들이 lung.py와 같은 폴더에 있거나 경로가 맞는지 확인해 주세요.")
    st.stop()
# ====================================================================

st.title("환자 군집 예측 및 시각화 프로그램")
st.write("환자의 정보를 입력하면 어떤 군집에 속하는지 예측하고 그래프에 표시합니다.")

st.divider()

## 1. 데이터 입력 (사이드바)
st.sidebar.header("환자 정보 입력")

Smokes = st.sidebar.number_input("흡연량 입력 (Smokes)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
AreaQ = st.sidebar.number_input("지역지수 입력 (AreaQ)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
Alkhol = st.sidebar.number_input("음주량 입력 (Alkhol)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# 분석 시작 버튼
if st.sidebar.button("군집 예측하기"):
    
    ## 2. 데이터프레임 변환 및 스케일링
    new_patient = pd.DataFrame([[Smokes, AreaQ, Alkhol]], columns=['Smokes', 'AreaQ', 'Alkhol'])
    
    # 이제 위에서 정상적으로 로드된 scaler와 model을 사용합니다.
    new_patient_scaled = scaler.transform(new_patient)
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
    
    # 기존 데이터의 군집 컬럼명이 'Result'라면 c=df['Result']로, 'cluster'라면 그대로 사용
    cluster_column = 'Result' if 'Result' in df.columns else 'cluster'
    
    scatter = ax.scatter(df['Smokes'], df['Alkhol'], c=df[cluster_column], alpha=0.5, cmap='viridis')
    
    # 새 환자 표시
    ax.scatter(Smokes, Alkhol, c='red', s=300, marker='X', label='New Patient')
    
    ax.set_xlabel('흡연량 (Smokes)')
    ax.set_ylabel('음주량 (Alkhol)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 스트림릿에 plot 표시
    st.pyplot(fig)