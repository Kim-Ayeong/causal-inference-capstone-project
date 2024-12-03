#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import locale
locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')
import datetime
import altair as alt

st.set_page_config(layout="wide")

df = pd.read_pickle('./rawdata.pickle')

# 사이드바 섹션
st.sidebar.title('Method Recommendation')

st.sidebar.header("1. 분석 모델 선택", divider=True)

# 세션 상태 초기화
if "show_recommendation" not in st.session_state:
    st.session_state.show_recommendation = False

# 추천 모델 출력
def show_recommendation_sidebar(method):
    st.sidebar.markdown(f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
        <h3 style="color: #4CAF50;">추천 모델:</h3>
        <p style="font-size: 16px;">{method}</p>
    </div>
    """, unsafe_allow_html=True)

# 추천 모델 숨김
def reset_recommendation():
    st.session_state.show_recommendation = False

# 첫 번째 질문
with st.sidebar:
    st.subheader("연구 설계 트리")

    # 연구 설계 여부
#     on = st.toggle("설명")
#     if on:
#         st.write("준실험 연구란? ~~")
    
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
                    recommendation = "이중차분법(Difference-in-Differences, DID)"
                else:
                    recommendation = "1) 메타러너(Meta learners) <br>2) 이중차분법(DID) + 매칭(Matching) 또는 통제집단합성법(Synthetic Control Method, SCM)"
            else:
                recommendation = "회귀 불연속(Regression Discontinuity, RD)"
        else:
            time_series = st.radio(
                "처치 전후의 시계열 데이터가 존재하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
            if time_series == "Yes":
                recommendation = "단절 시계열분석(Interrupted Time Series Analysis, ITS)"
            else:
                recommendation = "연구 설계 불가"
    else:
        instrumental_variable = st.radio(
            "도구변수가 존재하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)

        if instrumental_variable == "Yes":
            recommendation = "1) 2단계 최소제곱법(2-Stage Least Squares, 2SLS) <br>2) 선택편향 상관분석(Selection Bias Correction)"
        else:
            setting_clear = st.radio(
                "연구 설정이 명확하고 충분한 샘플이 존재하는가?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
            if setting_clear == "Yes":
                recommendation = "매칭(Matching)"
            else:
                recommendation = "회귀분석(Regression)"

# 추천 분석 방법 보기 버튼을 누르면 추천이 보이도록 설정
if st.sidebar.button("추천 모델 보기"):
    st.session_state.show_recommendation = True

# 추천 결과가 사이드바에 표시되도록 설정
if st.session_state.show_recommendation:
    show_recommendation_sidebar(recommendation)

#####
st.sidebar.header("2. 데이터 선택", divider=True)

choice_treat = st.sidebar.selectbox("처치(Treatment)에 해당하는 열을 선택하세요.", ['-']+list(df.columns))
choice_outcome = st.sidebar.selectbox("결과(Outcome)에 해당하는 열을 선택하세요.", ['-']+list(df.columns))

st.sidebar.write("처치(Treatment) : ", choice_treat)
st.sidebar.write("결과(Outcome) : ", choice_outcome)

#####
st.sidebar.header("3. 기간 선택", divider=True)

start_date = datetime.date(2022, 1, 1)
treat_date = datetime.date(2024, 1, 27)
end_date = datetime.date(2024, 7, 31)

# Date input with default value from 2022-01-01 to today's date
d = st.sidebar.date_input("분석 기간을 선택해주세요.",
    (start_date, end_date), 
    start_date, 
    end_date,
    format="YYYY/MM/DD",
)
d = st.sidebar.date_input("처치 발생 시점을 선택해주세요.", treat_date)
st.sidebar.divider()

if st.sidebar.button('submit'):
     st.success("선택한 조건으로 적용되었습니다.")

# 테이블 섹션
st.title('Causal Inference')
st.subheader('Rawdata')

st.dataframe(df.reindex(columns=['일자', '시도', '시군구', 
                                 '목적통행량', '총 인구 수', '방문자 수', '평균통행거리(km)', 
                                 '인구당 목적통행량', 'treated', 'post', 'w']),
             width=1200, height=150)

# 그래프 섹션
# daily
#st.line_chart(df_ts, x='일', y='목적통행량', color='시도')
df_ts = pd.DataFrame(df.groupby(['일자', '시도'])['인구당 목적통행량'].mean())
df_ts = df_ts.reset_index()
c_d = alt.Chart(df_ts, title='Daily chart').mark_line().encode(
     x='일자:T', y='인구당 목적통행량', color='시도')
c_d_line = alt.Chart(pd.DataFrame({'일자':['2024-02-01']})).mark_rule(
    color='black', strokeWidth=3, strokeDash=[2,2]).encode(
    x='일자:T')
c = (c_d + c_d_line).properties(width=1000, height=300)
st.altair_chart(c)

# weekly
df_sgg = df.groupby(['주차', '시도', '시군구'])['인구당 목적통행량'].sum()
df_ts2 = pd.DataFrame(df_sgg.groupby(['주차', '시도']).mean())
df_ts2 = df_ts2.reset_index()
df_ts2 = df_ts2.loc[df_ts2['주차'] != '202431'].copy()
c_w = alt.Chart(df_ts2, title='Weekly chart').mark_line().encode(
      x='주차', y='인구당 목적통행량', color='시도')
c_w_line = alt.Chart(pd.DataFrame({'주차':['202405']})).mark_rule(
    color='black', strokeWidth=3, strokeDash=[2,2]).encode(
    x='주차')
c = (c_w + c_w_line).properties(width=1000, height=300)
st.altair_chart(c)

st.divider()

# 모델 섹션
st.subheader('Selected Model')
st.text_input('')

st.divider()

# 결과 섹션
st.subheader('Result')
st.caption('※ 아래 수치는 모두 1개월을 기준으로 산출된 수치입니다.')

col1, col2 = st.columns([1,3])
with col1:
    st.button('대중교통 이용량')
with col2:
    val1 = st.text_input(label='val1', value='')

col1, col2 = st.columns([1,3])
with col1:
    st.button('대중교통 이용 증감률(%)')
with col2:
    val2 = st.text_input(label='val2', value='')

col1, col2 = st.columns([1,3])
with col1:
    st.button('이산화탄소 배출량')
with col2:
    val3 = st.text_input(label='val3', value='')


