#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import datetime
import altair as alt
from PIL import Image

st.set_page_config(layout="wide")

# 사이드바 섹션
#st.sidebar.title('Causal Inference Navigator')

st.sidebar.markdown("""
<h2 style="color: #4CAF50; font-size: 24px; font-weight: bold; border-bottom: 3px solid #4CAF50; padding-bottom: 5px;">
    Causal Inference Navigator
</h2>
""", unsafe_allow_html=True)

st.sidebar.header("About")
st.sidebar.markdown("""
Causal Inference Navigator는 인과 추론 연구를 돕기 위한 도구입니다. 
데이터를 업로드하고 분석 모델을 선택하여, 연구 설계 및 결과를 시각화합니다.
""")

############################### Section: Data
st.header('Upload file')
#st.write("데이터 파일을 업로드하세요.")


uploaded_file = st.file_uploader("Upload your file here", type=["csv", "xlsx"])
#if uploaded_file:
#    st.success("파일이 성공적으로 업로드되었습니다.")
#    df = pd.read_csv(uploaded_file)  # Assuming it's a CSV for simplicity
#    st.dataframe(df.head())

if uploaded_file:
    # 데이터 로드
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        data = None

    if data is not None:
        # About Dataset 섹션
        with st.container():
            st.markdown(
                """
                <div style="background-color:#f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #ddd;">
                    <h4 style="color: #333;">About Dataset</h4>
                    <ul style="line-height: 1.8; font-size: 14px;">
                        <li><strong>Filename:</strong> {}</li>
                        <li><strong>Size:</strong> {:.2f} MB</li>
                        <li><strong>Observations:</strong> {} rows</li>
                        <li><strong>Columns:</strong> {} columns</li>
                    </ul>
                </div>
                """.format(
                    uploaded_file.name,
                    uploaded_file.size / (1024 * 1024),
                    data.shape[0],
                    data.shape[1],
                ),
                unsafe_allow_html=True,
            )

        # Columns Overview 섹션
        st.markdown(
            """
            <div style="background-color:#f1f1f1; padding: 20px; border-radius: 8px; border: 1px solid #ccc; margin-top: 20px; font-size: 13px;">
                <h5 style="color: #555;">Columns Overview</h5>
                <div style="display: flex; justify-content: space-between; text-align: center;">
                    <div>
                        <p style="color: #007bff; font-size: 12px; margin-bottom: 5px;">Numeric Columns</p>
                        <p style="font-size: 18px; font-weight: bold;">{}</p>
                    </div>
                    <div>
                        <p style="color: #007bff; font-size: 12px; margin-bottom: 5px;">Categorical Columns</p>
                        <p style="font-size: 18px; font-weight: bold;">{}</p>
                    </div>
                    <div>
                        <p style="color: #007bff; font-size: 12px; margin-bottom: 5px;">Boolean Columns</p>
                        <p style="font-size: 18px; font-weight: bold;">{}</p>
                    </div>
                    <div>
                        <p style="color: #007bff; font-size: 12px; margin-bottom: 5px;">Date/Time Columns</p>
                        <p style="font-size: 18px; font-weight: bold;">{}</p>
                    </div>
                </div>
            </div>
            """.format(
                data.select_dtypes(include=["number"]).shape[1],
                data.select_dtypes(include=["object"]).shape[1],
                data.select_dtypes(include=["bool"]).shape[1],
                data.select_dtypes(include=["datetime64"]).shape[1],
            ),
            unsafe_allow_html=True,
        )

        # 두 섹션 사이 공간 추가
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Show Raw Data 섹션
        with st.expander("Show Raw Data"):
            st.subheader("Raw Data")
            st.dataframe(data)

else:
    st.info("Please upload a file to see the dataset summary.")



###############################################

st.header('Model Selector')

# 세션 상태 초기화
if "show_recommendation" not in st.session_state:
    st.session_state.show_recommendation = False

# 추천 모델 출력
def show_recommendation(method):
    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
        <h3 style="color: #4CAF50;">추천 방법:</h3>
        <p style="font-size: 16px;">{method}</p>
    </div>
    """, unsafe_allow_html=True)

# 추천 모델 숨김
def reset_recommendation():
    st.session_state.show_recommendation = False

# 첫 번째 질문
st.subheader("Flowchart for Causal Inference")

# 연구 설계 여부
experimental_design = st.radio(
    "준실험 연구 설계가 가능한가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)

if experimental_design == "Yes":
    treatment_control = st.radio(
        "처치(Treatment)와 통제(Control) 그룹이 관찰되는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)

    if treatment_control == "Yes":
        panel_data = st.radio(
            "패널 데이터가 존재하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)

        if panel_data == "Yes":
            parallel_trends = st.radio(
                "평행 추세 가정이 성립하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
            if parallel_trends == "Yes":
                recommendation = "이중차분법 (Difference-in-Differences, DID)"
            else:
                recommendation = "DID + Matching 또는 합성 통제"
        else:
            recommendation = "회귀불연속 (Regression Discontinuity)"
    else:
        time_series = st.radio(
            "처치 전후의 시계열 데이터가 존재하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
        if time_series == "Yes":
            recommendation = "시계열 분석 (Interrupted Time-Series Analysis)"
        else:
            recommendation = "연구 설계 불가"
else:
    instrumental_variable = st.radio(
        "도구변수가 존재하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)

    if instrumental_variable == "Yes":
        recommendation = "Two-Stage Least Squares 또는 Selection Bias Correction"
    else:
        setting_clear = st.radio(
            "연구 설정이 명확하고 충분한 샘플이 존재하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
        if setting_clear == "Yes":
            recommendation = "Matching"
        else:
            recommendation = "Regression"

# 추천 분석 방법 보기 버튼을 누르면 추천이 보이도록 설정
if st.button("추천 분석방법 보기"):
    st.session_state.show_recommendation = True

# 추천 결과가 본문에 표시되도록 설정
if st.session_state.show_recommendation:
    show_recommendation(recommendation)



##############################################

st.header('Analysis')

# 데이터 선택 섹션
st.subheader("1. Choose Variables")

# 데이터프레임이 이미 업로드된 상태에서 사용
if 'data' in locals() and data is not None:
    # 칼럼 이름을 선택하도록 설정
    treat_column = st.selectbox("처치 데이터에 해당하는 열 이름을 선택하세요", options=data.columns, key="treat_column")
    outcome_column = st.selectbox("결과 데이터에 해당하는 열 이름을 선택하세요", options=data.columns, key="outcome_column")

    # 선택한 칼럼 표시
    st.write(f"**처치(열):** {treat_column}")
    st.write(f"**결과(열):** {outcome_column}")
else:
    st.warning("데이터가 로드되지 않았습니다. 먼저 파일을 업로드하세요.")

# 섹션 간 간격 추가
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)


# 기간 선택 섹션
st.subheader("2. Setup Date")

# 데이터 업로드 확인
if 'data' not in locals() or data is None:
    st.warning("데이터가 로드되지 않았습니다. 먼저 파일을 업로드하세요.")
else:
    # 분석 기간 선택
    st.write("분석을 희망하는 기간을 선택해주세요")
    analysis_period = st.date_input(
        "기간 선택:",
        value=(data['일자'].min(), data['일자'].max()),  # 데이터의 최소~최대 기간으로 초기값 설정
        min_value=data['일자'].min(),  # 데이터의 최소 날짜
        max_value=data['일자'].max(),  # 데이터의 최대 날짜
        format="MM.DD.YYYY",
    )

    # 처치 발생 시점 선택
    treatment_date = st.date_input(
        "처치가 발생한 시점을 선택해주세요",
        value=data['일자'].max(),  # 기본값은 데이터의 최대 날짜
        min_value=data['일자'].min(),
        max_value=data['일자'].max(),
    )

    # 제출 버튼
    if st.button("Submit"):
        # 선택된 기간에 맞춰 데이터 필터링
        filtered_data = data[
            (data['일자'] >= pd.to_datetime(analysis_period[0])) &
            (data['일자'] <= pd.to_datetime(analysis_period[1]))
        ]

        # 일별 차트 생성
        df_daily = (
            filtered_data.groupby(['일자', '시도'])['인구당 목적통행량']
            .mean()
            .reset_index()
        )
        daily_chart = alt.Chart(df_daily, title='Daily Chart').mark_line().encode(
            x='일자:T',
            y='인구당 목적통행량',
            color='시도'
        )
        treatment_line = alt.Chart(pd.DataFrame({'일자': [treatment_date]})).mark_rule(
            color='black', strokeWidth=3, strokeDash=[2, 2]
        ).encode(x='일자:T')
        daily_combined = (daily_chart + treatment_line).properties(width=1000, height=300)

        # 주별 차트 생성
        filtered_data['주차'] = filtered_data['일자'].dt.strftime('%Y%U')  # 주차 계산
        df_weekly = (
            filtered_data.groupby(['주차', '시도'])['인구당 목적통행량']
            .mean()
            .reset_index()
        )
        weekly_chart = alt.Chart(df_weekly, title='Weekly Chart').mark_line().encode(
            x='주차:O',
            y='인구당 목적통행량',
            color='시도'
        )
        treatment_week = alt.Chart(pd.DataFrame({'주차': [treatment_date.strftime('%Y%U')]})).mark_rule(
            color='black', strokeWidth=3, strokeDash=[2, 2]
        ).encode(x='주차:O')
        weekly_combined = (weekly_chart + treatment_week).properties(width=1000, height=300)

        # 차트 출력
        st.altair_chart(daily_combined, use_container_width=True)
        st.altair_chart(weekly_combined, use_container_width=True)

st.button("추천 분석방법 적용하기")

##############################################

# Result 섹션

st.header('Results')


# 1. 효과를 직관적으로 숫자로 보여주는 섹션
st.subheader("Effect Summary")
col1, col2, col3 = st.columns(3)
col1.metric("처치 이전", "00.0%")
col2.metric("처치 이후", "00.0%", "+0.0%")

# 섹션 간 간격 추가
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

# 2. 그래프 섹션 (선 그래프, BOX PLOT, 회귀)
st.subheader("Visual Analysis")

###예시 그래프
# 선 그래프
st.write("### Line Chart")
line_chart_data = pd.DataFrame({
    "날짜": pd.date_range(start="2023-01-01", periods=100),
    "효과": [i * 0.1 for i in range(100)]
})
line_chart = alt.Chart(line_chart_data).mark_line().encode(
    x="날짜:T",
    y="효과:Q"
)
st.altair_chart(line_chart, use_container_width=True)

# BOX PLOT
st.write("### Box Plot")
box_plot_data = pd.DataFrame({
    "그룹": ["처치", "통제"] * 50,
    "값": [i * 0.1 + (0 if i % 2 == 0 else 5) for i in range(100)]
})
box_plot = alt.Chart(box_plot_data).mark_boxplot().encode(
    x="그룹:N",
    y="값:Q"
)
st.altair_chart(box_plot, use_container_width=True)

# 섹션 간 간격 추가
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

# 3. 결과를 서술형으로 문장 작성
st.subheader("Narrative Summary")
result_variable = "효과"  # 예시 변수
change_percentage = "+0.0%"  # 예시 값
st.write(f"""
처치 이후 **{result_variable}**에 대한 변화는 **{change_percentage}**로 분석되었습니다.
이 결과는 처치가 효과적임을 시사하며, 추가적인 분석에서 유의미한 결과가 도출될 수 있습니다.
""")

# 섹션 간 간격 추가
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)

# 4. PDF 리포트 저장 버튼
st.subheader("Export Report")
if st.button("Download PDF"):
    st.success("PDF 리포트가 성공적으로 저장되었습니다.")
    # PDF 생성 코드는 필요시 추가 가능

