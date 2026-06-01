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

## 1. 데이터 입력 (사이드바) - [Age(나이) 추가 완료!]
st.sidebar.header("환자 정보 입력")

Age = st.sidebar.number_input("나이 입력 (Age)", min_value=0, max_value=120, value=30, step=1)
Smokes = st.sidebar.number_input("흡연량 입력 (Smokes)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
AreaQ = st.sidebar.number_input("지역지수 입력 (AreaQ)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
Alkhol = st.sidebar.number_input("음주량 입력 (Alkhol)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

# 분석 시작 버튼
if st.sidebar.button("군집 예측하기"):
    
    ## 2. 데이터프레임 변환 및 스케일링 (학습 때와 동일한 4개 컬럼 구성)
    # 원래 학습에 사용된 컬럼 순서인 ['Age', 'Smokes', 'AreaQ', 'Alkhol']을 강제로 맞춥니다.
    new_patient = pd.DataFrame([[Age, Smokes, AreaQ, Alkhol]], columns=['Age', 'Smokes', 'AreaQ', 'Alkhol'])
    
    # 순서와 개수가 일치하므로 이제 에러 없이 변환 및 예측이 됩니다!
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
