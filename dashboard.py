#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import datetime
import altair as alt
from PIL import Image
import causal_function as cf
from fpdf import FPDF

st.set_page_config(layout="wide")

# session state 유지
if "selected_data_type" not in st.session_state:
    st.session_state["selected_data_type"] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

###################################
# Sidebar section
###################################

st.sidebar.markdown("""
<h2 style="color: #4CAF50; font-size: 24px; font-weight: bold; border-bottom: 3px solid #4CAF50; padding-bottom: 5px;">
    Causal Inference Navigator
</h2>
""", unsafe_allow_html=True)

st.sidebar.header("| **About**")
st.sidebar.markdown("""
Causal Inference Navigator는 누구나 쉽게 인과 추론을 할 수 있도록 설계된 도구입니다. \n
데이터 업로드 후, 분석 모델 및 필요한 정보를 입력하면 인과 효과를 분석해 시각화합니다. \n
이 도구는 복잡한 데이터로부터 인과적인 통찰을 얻고, 데이터 기반의 의사결정을 내릴 수 있도록 지원합니다.
""")


# 데이터 유형 선택
st.sidebar.divider()

st.markdown("""
    <style>
    .stButton > button {
        display: block;
        width: 90%;
        margin: 10px auto;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        background-color: #28a745;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #218838;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("| **Select Data Type**")
st.sidebar.markdown("데이터 유형을 선택해주세요.")

# 세션 반영
if st.sidebar.button("실험 데이터", key="exp_data_button"):
    st.session_state["selected_data_type"] = "실험 데이터"

if st.sidebar.button("관찰 데이터", key="obs_data_button"):
    st.session_state["selected_data_type"] = "관찰 데이터"
    
# 결과 초기값
result_flag = False

treat_column = ''
outcome_column = ''

data_column = ''
start_date = ''
end_date = ''
treat_date = ''

pre_treat = ''
change_coef = ''
post_treat = ''
change_perc = ''
change_flag = ''

p_value = ''
stats_sign = ''

###################################
# exp data section
###################################
if st.session_state["selected_data_type"] == "실험 데이터":
    
    st.header('Upload File')
    
    uploaded_file = st.file_uploader("", type=["csv", "xlsx"])
    if uploaded_file:
        st.success("파일이 성공적으로 업로드되었습니다.")
        
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

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
            # Show Data 섹션
            with st.expander("Show Data"):
                st.dataframe(data)
#     else:
#         st.info("Please upload the data first.")
    
    st.divider()
    ##############################################
    
    st.header('Select Variables')
    
    st.subheader("1) Treatment, Outcome")
    
    if 'data' in locals() and data is not None:
        treat_column = st.selectbox("처치(Treatment)에 해당하는 열을 선택하세요.", options=data.columns, key="treat_column")
        outcome_column = st.selectbox("결과(Outcome)에 해당하는 열을 선택하세요.", options=data.columns, key="outcome_column")
    
        st.write(f"**처치(Treatment):** {treat_column}")
        st.write(f"**결과(Outcome):** {outcome_column}")
        
    else:
        st.warning("Please upload the data first.")
        
    if st.button("Submit"):
        st.success("선택한 정보가 성공적으로 적용되었습니다.")
        result_flag = True
        
        # function result
        result = cf.ab_test(data, treat_column, outcome_column)
        pre_treat = str(result['pre_treat'])
        change_coef = str(result['change_coef'])
        post_treat = str(result['post_treat'])
        change_perc = str(result['change_perc']) + '%'
        if result['change_coef'] >= 0:
            change_flag = '+'
        p_value = str(result['p_value'])
        if result['p_value'] <= 0.05:
            stats_sign = 'True'
        else:
            stats_sign = 'False'
    
    st.divider()
    ##############################################
    
    st.header('Results')
    
    # 1. 숫자 섹션
    st.subheader("1) Effect Summary")
    
    if uploaded_file:
        if result_flag:
            col1, col2, col3 = st.columns(3)
            col1.metric("처치 전", f"{pre_treat}")
            col2.metric("처치 후", f"{post_treat}", f"{change_flag + change_perc}")
        else:
            st.warning("Please select variables next.")
    else:
        st.warning("Please upload the data first.")

    
    # 2. 그래프 섹션
    st.subheader("2) Effect Visualization")
    
    if uploaded_file:
        if result_flag:
            if data[outcome_column].dtype in [object, bool]:
                if data[outcome_column].dtype == object:
                    data[outcome_column] = data[outcome_column].map(lambda x: 1 if str(x).lower() == 'true' else 0)
                else:
                    data[outcome_column] = data[outcome_column].astype(int)

                st.write("#### Bar Plot")
                bar_plot = alt.Chart(data).mark_bar(size=100).encode(
                    x=alt.X(f'{treat_column}:N'),
                    y=alt.Y(f'mean({outcome_column}):Q'),
                )
                st.altair_chart(bar_plot, use_container_width=True)

            else:
                st.write("#### Box Plot")
                
                box_plot = alt.Chart(data).mark_boxplot(size=100).encode(
                    x=treat_column,
                    y=outcome_column
                )
                st.altair_chart(box_plot, use_container_width=True)
        else:
            st.warning("Please select variables next.")
    else:
        st.warning("Please upload the data first.")
    
    # 3. 서술형 요약
    st.subheader("3) Narrative Summary")
    
    if uploaded_file:
        if result_flag:
            st.write(f"""
            **{treat_column}**에 따른 **{outcome_column}** 효과는 {change_flag + change_perc}로 분석되었습니다. \n
            이 때, p-value는 **{p_value}**로 나타났으며 유의수준 5% 내에서 통계적 유의성은 **{stats_sign}**입니다. \n
            위 결과는 **{outcome_column}**에 대한 추가 분석을 거쳐 인과적 근거로 활용될 수 있습니다.
            """)
        else:
            st.warning("Please select variables next.")
    else:
        st.warning("Please upload the data first.")

    st.divider()
    ##############################################
    
    # PDF 리포트 저장 버튼
    st.header("Export Report")
    
    def generate_pdf_narrative(treat_column, outcome_column, change_flag, change_perc, p_value, stats_sign):
        """Generate a PDF containing the narrative summary."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # 한글 폰트 추가 (맑은 고딕 또는 나눔고딕 사용)
#        font_path = "/path/to/your/font.ttf"  # 한글 폰트 파일 경로 (예: NanumGothic.ttf)
#        pdf.add_font("NanumGothic", fname=font_path, uni=True)
#        pdf.set_font("NanumGothic", size=12)
        
        # Add narrative summary content
        pdf.cell(200, 10, txt="Narrative Summary", ln=True, align='C')
        pdf.ln(10)
        narrative_text = (
            f"The effect of **{treat_column}** on **{outcome_column}** was analyzed as {change_flag + change_perc}.\n"
            f"The p-value was found to be **{p_value}**, and it was statistically significant at the 5% level (**{stats_sign}**).\n"
            f"This result can be used as causal evidence for further analysis of **{outcome_column}**."
        )
        pdf.multi_cell(0, 10, narrative_text)
        
        # Save the PDF file
        file_path = "narrative_summary.pdf"
        pdf.output(file_path)
        return file_path


    
    if st.button("Download PDF"):
        pdf_file_path = generate_pdf_narrative(treat_column, outcome_column, change_flag, change_perc, p_value, stats_sign)
        st.success("PDF가 성공적으로 저장되었습니다.")

        # 바로 다운로드 제공
        with open(pdf_file_path, "rb") as pdf_file:
            st.download_button(
                label="Click here to download the PDF",
                data=pdf_file,
                file_name="narrative_summary.pdf",
                mime="application/pdf"
            )

###################################
# obs data section
###################################

elif st.session_state["selected_data_type"] == "관찰 데이터":

    st.header('Upload File')
    
    uploaded_file = st.file_uploader("", type=["csv", "xlsx"])
    if uploaded_file:
        st.success("파일이 성공적으로 업로드되었습니다.")
        
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

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
            # Show Data 섹션
            with st.expander("Show Data"):
                st.dataframe(data)
#     else:
#         st.info("Please upload the data first.")
    
    st.divider()
    ###############################################
    
    st.header('Causal Model Selector')
    
    # 세션 상태 초기화
    if "show_recommendation" not in st.session_state:
        st.session_state.show_recommendation = False
    
    # 추천 모델 출력
    def show_recommendation(model):
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
            <p style="font-size: 16px;">{model}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 추천 모델 숨김
    def reset_recommendation():
        st.session_state.show_recommendation = False
    
    # flowchart
    with st.expander("Flowchart for Causal Inference"):
        st.markdown("<div style='height: 20px;'> ~ flowchart image ~ </div>", unsafe_allow_html=True)
    
    # process
    experimental_design = st.radio(
        "준실험 연구 설계가 가능한가요?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
    
    if experimental_design == "Yes":
        treatment_control = st.radio(
            "처치(Treatment), 통제(Control) 그룹을 정의할 수 있나요?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
    
        if treatment_control == "Yes":
            panel_data = st.radio(
                "같은 대상이 시간별로 기록된 패널 데이터인가요?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
    
            if panel_data == "Yes":
                parallel_trends = st.radio(
                    "평행 추세 가정이 성립하나요?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
                if parallel_trends == "Yes":
                    recommendation = "이중차분법 (Difference-in-Differences, DID)"
                else:
                    recommendation = "이중차분법 (Difference-in-Differences, DID) + 매칭 (Matching) 또는 합성통제 (Synthetic Control)"
            else:
                recommendation = "회귀불연속 (Regression Discontinuity)"
        else:
            time_series = st.radio(
                "처치 전후의 시계열 데이터가 존재하나요?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
            if time_series == "Yes":
                recommendation = "단절 시계열분석 (Interrupted Time Series Analysis)"
            else:
                recommendation = "연구 설계 불가"
    else:
        instrumental_variable = st.radio(
            "도구변수가 존재하나요?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
    
        if instrumental_variable == "Yes":
            recommendation = "2단계 최소제곱법 (2-Stage Least Squares)"
        else:
            setting_clear = st.radio(
                "처치 및 결과가 명확하고, 각 그룹에 최소 30개의 데이터가 있나요?", ["Yes", "No"], horizontal=True, on_change=reset_recommendation)
            if setting_clear == "Yes":
                recommendation = "매칭 (Matching)"
            else:
                recommendation = "회귀분석 (Regression)"
    
    if st.button("추천 모델 확인"):
        st.session_state.show_recommendation = True
    
    # 추천 결과 세션
    if st.session_state.show_recommendation:
        show_recommendation(recommendation)
    
    st.divider()
    ##############################################
    st.header('Select Model')
    selected_model = st.selectbox("인과추론 모델을 선택하세요.", options=['DID', 'RD'], key="selected_model")
    
    if st.button("추천 모델 적용"):
        st.success("선택한 모델이 성공적으로 적용되었습니다.")

    # 추천 분석방법 적용하기 버튼
#     if st.session_state["selected_data_type"] and "uploaded_file" in st.session_state and st.session_state["uploaded_file"]:
#         if st.button("추천 분석방법 적용하기"):
#             st.write("추천 분석방법 적용하기 버튼이 클릭되었습니다.")
#             run_did_analysis(st.session_state["uploaded_file"])
#     else:
#         st.info("Please upload the data first.")

    
    st.divider()
    ##############################################
    
    st.header('Select Variables')
    
    st.subheader("1) Treatment, Outcome")
    
    if 'data' in locals() and data is not None:
        treat_column = st.selectbox("처치(Treatment)에 해당하는 열을 선택하세요.", options=data.columns, key="treat_column")
        outcome_column = st.selectbox("결과(Outcome)에 해당하는 열을 선택하세요.", options=data.columns, key="outcome_column")
    
        st.write(f"**처치(Treatment):** {treat_column}")
        st.write(f"**결과(Outcome):** {outcome_column}")
    else:
        st.warning("Please upload the data first.")
        
#     st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)
    
    st.subheader("2) Date")
    
    if 'data' not in locals() or data is None:
        st.warning("Please upload the data first.")
    else:
        date_column = st.selectbox("날짜에 해당하는 열을 선택하세요.", options=data.columns, key="date_column")
        try:
            analysis_period = st.date_input(
                "분석에 사용할 기간을 선택해주세요.",
                value = (data[date_column].min(), data[date_column].max()),  # 최소~최대 기간
                min_value = data[date_column].min(),  # 최소 날짜
                max_value = data[date_column].max(),  # 최대 날짜
                format = "YYYY/MM/DD"
            )

            treat_date = st.date_input(
                "처치(Treatment)가 발생한 시점을 선택해주세요.",
                value = data[date_column].max(),  # 최대 날짜
                min_value = data[date_column].min(),
                max_value = data[date_column].max(),
                format = "YYYY/MM/DD"
            )
        except:
            pass
        
    if st.button("Submit"):
        st.success("선택한 정보가 성공적으로 적용되었습니다.")
        result_flag = True

        start_date = analysis_period[0]
        end_date = analysis_period[1]

        # function result
        if selected_model == 'DID':
            result = cf.did(data, treat_column, outcome_column, start_date, end_date, treat_date)
        
        elif selected_model == 'RD':
            result = cf.rd(data, treat_column, outcome_column, start_date, end_date, treat_date)
        
        else:
            pass
        
        pre_treat = str(result['pre_treat'])
        change_coef = str(result['change_coef'])
        post_treat = str(result['post_treat'])
        change_perc = str(result['change_perc']) + '%'
        if result['change_coef'] >= 0:
            change_flag = '+'
        p_value = str(result['p_value'])
        if result['p_value'] <= 0.05:
            stats_sign = 'True'
        else:
            stats_sign = 'False'

        
    st.divider()
    ##############################################
    
    st.header('Results')
    
    # 1. 숫자 섹션
    st.subheader("1) Effect Summary")
    
    if uploaded_file:
        if result_flag:
            col1, col2, col3 = st.columns(3)
            col1.metric("처치 전", f"{pre_treat}")
            col2.metric("처치 후", f"{post_treat}", f"{change_flag + change_perc}")
        else:
            st.warning("Please select variables next.")
    else:
        st.warning("Please upload the data first.")

    
    # 2. 그래프 섹션
    st.subheader("2) Effect Visualization")
    
    if uploaded_file:
        if result_flag:

                #     # 주별 차트 생성
                #     filtered_data['주차'] = filtered_data['일자'].dt.strftime('%Y%U')  # 주차 계산
                #     df_weekly = (
                #         filtered_data.groupby(['주차', '시도'])['인구당 목적통행량']
                #         .mean()
                #         .reset_index()
                #     )
                #     weekly_chart = alt.Chart(df_weekly, title='Weekly Chart').mark_line().encode(
                #         x='주차:O',
                #         y='인구당 목적통행량',
                #         color='시도'
                #     )
                #     treatment_week = alt.Chart(pd.DataFrame({'주차': [treatment_date.strftime('%Y%U')]})).mark_rule(
                #         color='black', strokeWidth=3, strokeDash=[2, 2]
                #     ).encode(x='주차:O')
                #     weekly_combined = (weekly_chart + treatment_week).properties(width=1000, height=300)

                #     # 차트 출력
                #     st.altair_chart(weekly_combined, use_container_width=True)
            
            # 선 그래프
            st.write("#### Line Chart")
            line_chart = alt.Chart(data).mark_line().encode(
                x = date_column,
                y = outcome_column,
                color = treat_column
            )
            treatment_line = alt.Chart(pd.DataFrame({date_column: [treat_date]})).mark_rule(
                color='black', strokeWidth=3, strokeDash=[2, 2]
            ).encode(x=f'{date_column}:T')
            
            combined_chart = line_chart + treatment_line
            st.altair_chart(combined_chart, use_container_width=True)
            
            
            st.write("#### Box Plot")
            box_plot = alt.Chart(data).mark_boxplot(size=100).encode(
                x = treat_column,
                y = outcome_column
            )
            st.altair_chart(box_plot, use_container_width=True)
            
        else:
            st.warning("Please select variables next.")
    else:
        st.warning("Please upload the data first.")
    
    # 3. 서술형 요약
    st.subheader("3) Narrative Summary")
    
    if uploaded_file:
        if result_flag:
            st.write(f"""
            **{start_date}**부터 **{end_date}**까지 데이터를 활용해 **{treat_date}** 발생한 **{treat_column}** 효과를 분석하였습니다. \n
            **{treat_column}**에 따른 **{outcome_column}** 효과를 분석한 결과, {change_flag + change_perc}로 분석되었습니다. \n
            이 때, p-value는 **{p_value}**로 나타났으며 유의수준 5% 내에서 통계적 유의성은 **{stats_sign}**입니다. \n
            위 결과는 **{outcome_column}**에 대한 추가 분석을 거쳐 인과적 근거로 활용될 수 있습니다.
            """)
            
        else:
            st.warning("Please select variables next.")
    else:
        st.warning("Please upload the data first.")
    
#     # 섹션 간 간격 추가
#     st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)
    st.divider()
    ##############################################
    
    # PDF 리포트 저장 버튼
    st.header("Export Report")
    
    if st.button("Download PDF"):
        st.success("PDF 리포트가 성공적으로 저장되었습니다.")    


    def generate_pdf():
        """Generate an example pdf file and save it to example.pdf"""
        from fpdf import FPDF
    
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # Add content
        pdf.cell(200, 10, txt="Effect Summary", ln=True, align='C')
        pdf.ln(10)
#        pdf.cell(200, 10, txt=f"처치 전: {pre_treat}", ln=True)
#        pdf.cell(200, 10, txt=f"처치 후: {post_treat}", ln=True)
#        pdf.cell(200, 10, txt=f"변화율: {change_flag + change_perc}%", ln=True)
        pdf.ln(10)
#        pdf.cell(200, 10, txt=f"p-value: {p_value}", ln=True)
#        pdf.cell(200, 10, txt=f"통계적 유의성: {stats_sign}", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt="위 결과는 목적통행량에 대한 추가 분석을 거쳐 인과적 근거로 활용될 수 있습니다.", ln=True)     
        pdf.output("example.pdf")
    
    
    if st.button("Generate PDF"):
        generate_pdf()
        st.success("Generated example.pdf!")
    
    with open("example.pdf", "rb") as f:
        st.download_button("Download pdf", f, "example.pdf")
    


    ######################
#     import matplotlib.pyplot as plt
#     from io import BytesIO
#     from reportlab.pdfgen import canvas
    
#     # 예시 차트 생성
#     def create_chart():
#         fig, ax = plt.subplots()
#         ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
#         ax.set_title("Line Chart Example")
#         return fig
    
#     # PDF 생성 함수
#     def create_pdf_with_results(fig, results_text):
#         buf = BytesIO()
#         c = canvas.Canvas(buf)
        
#         # 텍스트 삽입
#         c.setFont("Helvetica-Bold", 16)
#         c.drawString(100, 750, "Results Summary")
        
#         # Effect Summary (예시)
#         c.setFont("Helvetica", 12)
#         c.drawString(100, 730, "Effect Summary")
#         c.drawString(100, 710, f"처치 이전: {results_text['before']}")
#         c.drawString(100, 690, f"처치 이후: {results_text['after']}")
#         c.drawString(100, 670, f"변화: {results_text['change']}")
        
#         # 차트를 이미지로 저장하고 PDF에 삽입
#         fig.savefig("/mnt/data/temp_chart.png")
#         c.drawImage("/mnt/data/temp_chart.png", 100, 400, width=400, height=300)
        
#         c.save()
#         buf.seek(0)
#         return buf
    
#     # 스트림릿 대시보드
#     st.subheader("Export Results as PDF")
    
#     results_text = {
#         'before': '00.0%',
#         'after': '00.0%',
#         'change': '+0.0%'
#     }
    
#     if st.button("Download PDF"):
#         fig = create_chart()  # 차트 생성
#         pdf_buf = create_pdf_with_results(fig, results_text)  # PDF 생성
#         st.download_button("Download PDF Report", pdf_buf, "results_report.pdf", mime="application/pdf")
#         st.success("PDF 리포트가 성공적으로 저장되었습니다.")


