import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 데이터 로드 및 탐색
def load_and_explore_data(file_path):
    """데이터 로드 및 기본 탐색"""
    df = pd.read_csv(file_path)
    
    # 컬럼명 공백을 언더스코어로 변경
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    print("=== 데이터 기본 정보 ===")
    print(df.info())
    print("\n=== 결측치 확인 ===")
    print(df.isnull().sum())
    print("\n=== 기술 통계량 ===")
    print(df.describe())
    
    # 그룹별 데이터 크기 확인
    group_sizes = df['test_group'].value_counts()
    print("\n=== 그룹별 데이터 크기 ===")
    print(group_sizes)
    
    return df

# 2. 데이터 전처리
def preprocess_data(df):
    """데이터 전처리 및 샘플링"""
    # Control 그룹과 Treatment 그룹의 크기 확인
    control_size = len(df[df['test_group'] == 'psa'])
    treatment_size = len(df[df['test_group'] == 'ad'])
    print(f"\n=== 원본 데이터 그룹 크기 ===")
    print(f"Control(psa) 그룹 크기: {control_size}")
    print(f"Treatment(ad) 그룹 크기: {treatment_size}")
    
    # Control 그룹은 전체 데이터 사용
    control_data = df[df['test_group'] == 'psa']
    # Treatment 그룹에서 Control 그룹 크기의 2배 정도로 샘플링
    sample_size = min(treatment_size, control_size * 2)
    treatment_data = df[df['test_group'] == 'ad'].sample(n=sample_size, random_state=42)
    
    print(f"\n=== 처리 후 데이터 그룹 크기 ===")
    print(f"Control(psa) 그룹 크기: {len(control_data)}")
    print(f"Treatment(ad) 그룹 크기: {len(treatment_data)}")
    
    # 데이터 병합
    processed_df = pd.concat([control_data, treatment_data])
    
    # 범주형 변수 더미화
    processed_df = pd.get_dummies(processed_df, columns=['most_ads_day'], drop_first=True)
    processed_df['test_group_binary'] = (processed_df['test_group'] == 'ad').astype(int)
    
    return processed_df

def calculate_smd(df, var, treatment_col):
    """표준화된 평균 차이(SMD) 계산"""
    treat = df[df[treatment_col] == 1][var]
    control = df[df[treatment_col] == 0][var]
    
    diff = treat.mean() - control.mean()
    pooled_std = np.sqrt((treat.var() + control.var()) / 2)
    
    return diff / pooled_std if pooled_std != 0 else 0

def visualize_distributions(df_before, df_after, var, treatment_col):
    """매칭 전후 연속형 변수 분포 시각화"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Before Matching', 'After Matching'))
    
    # Before Matching
    for i, group in enumerate([0, 1]):
        data = df_before[df_before[treatment_col] == group][var]
        fig.add_trace(
            go.Histogram(x=data, name=f'Group {group}', opacity=0.7),
            row=1, col=1
        )
    
    # After Matching
    for i, group in enumerate([0, 1]):
        data = df_after[df_after[treatment_col] == group][var]
        fig.add_trace(
            go.Histogram(x=data, name=f'Group {group}', opacity=0.7),
            row=1, col=2
        )
    
    fig.update_layout(
        title=f'Distribution of {var} Before and After Matching',
        barmode='overlay'
    )
    fig.show()

def evaluate_matching_quality(df_before, df_after, treatment_col, variables):
    """매칭 품질 평가 및 시각화 (개선된 버전)"""
    # 기존 SMD 계산 및 Love plot
    smd_before = {var: calculate_smd(df_before, var, treatment_col) for var in variables}
    smd_after = {var: calculate_smd(df_after, var, treatment_col) for var in variables}
    
    # Love plot (기존 코드 유지)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(smd_before.values()),
        y=list(smd_before.keys()),
        name='Before Matching',
        mode='markers',
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(smd_after.values()),
        y=list(smd_after.keys()),
        name='After Matching',
        mode='markers',
        marker=dict(size=10)
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.1, line_dash="dash", line_color="red")
    fig.add_vline(x=-0.1, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='Standardized Mean Differences Before and After Matching',
        xaxis_title='Standardized Mean Difference',
        showlegend=True
    )
    
    fig.show()
    
    # 연속형 수에 대한 분포 비교
    continuous_vars = ['most_ads_hour']  # 필요한 연속형 변수 추가
    for var in continuous_vars:
        visualize_distributions(df_before, df_after, var, treatment_col)
    
    return smd_before, smd_after

# 3. Propensity Score Matching
def perform_psm(df):
    """개선된 프로펜시티 스코어 매칭 수행"""
    # 매칭 변수 설정 (더미화된 변수들 포함)
    matching_vars = ['most_ads_hour'] + [col for col in df.columns if col.startswith('most_ads_day_')]
    X = df[matching_vars]
    treatment = df['test_group_binary']
    
    # Propensity Score 계산
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, treatment)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    
    # 매칭 수행 (caliper 추가)
    caliper = 0.2 * np.std(propensity_scores)
    nn = NearestNeighbors(n_neighbors=1)
    
    treatment_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    
    # propensity score를 2D 배열로 변환
    control_ps = propensity_scores[control_idx].reshape(-1, 1)
    treatment_ps = propensity_scores[treatment_idx].reshape(-1, 1)
    
    nn.fit(control_ps)
    distances, matches = nn.kneighbors(treatment_ps)
    
    # caliper를 적용하여 매칭 필터링
    good_matches = distances.flatten() <= caliper
    
    matched_treatment_idx = treatment_idx[good_matches]
    matched_control_idx = control_idx[matches.flatten()[good_matches]]
    
    matched_df = pd.concat([
        df.iloc[matched_treatment_idx],
        df.iloc[matched_control_idx]
    ])
    
    # 매칭 품질 평가
    print("\n=== 매칭 품질 평가 ===")
    evaluate_matching_quality(
        df, matched_df, 
        'test_group_binary',
        matching_vars
    )
    
    return matched_df

# 4. Logit 분석
def perform_logit_analysis(df):
    """개선된 로지스틱 회귀 분석 수행"""
    # 상호작용항 추가
    X = df[['test_group_binary', 'total_ads', 'most_ads_hour']]
    X['ads_treatment_interaction'] = X['total_ads'] * X['test_group_binary']
    X = sm.add_constant(X)
    y = df['converted']
    
    # VIF 계산
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\n=== VIF 분석 결과 ===")
    print(vif_data)
    
    # 로지스틱 회귀 분석
    model = sm.Logit(y, X)
    results = model.fit()
    
    # 변수 중요도 계산
    importance = pd.DataFrame({
        'variable': X.columns,
        'coefficient': results.params,
        'std_err': results.bse,
        'p_value': results.pvalues
    })
    importance['abs_importance'] = abs(importance['coefficient'])
    importance = importance.sort_values('abs_importance', ascending=False)
    
    print("\n=== 변수 중요도 ===")
    print(importance)
    
    print("\n=== Logit 분석 결과 ===")
    print(results.summary())
    
    return results, vif_data, importance

# 5. 비즈니스 인사이트 도출
def calculate_insights(df, matched_df):
    """비즈니스 인사이트 계산 및 시각화 (개선된 버전)"""
    # 전환율 계산 (그룹명 수정)
    def calc_conversion_rate(data, group):
        group_data = data[data['test_group'] == group]
        return group_data['converted'].mean()
    
    # 전환율 계산
    original_control_cr = calc_conversion_rate(df, 'psa')
    original_treatment_cr = calc_conversion_rate(df, 'ad')
    matched_control_cr = calc_conversion_rate(matched_df, 'psa')
    matched_treatment_cr = calc_conversion_rate(matched_df, 'ad')
    
    # ATE 계산
    ate = matched_treatment_cr - matched_control_cr
    
    print("\n=== 전환율 분석 ===")
    print(f"Original Control CR: {original_control_cr:.4f}")
    print(f"Original Treatment CR: {original_treatment_cr:.4f}")
    print(f"Matched Control CR: {matched_control_cr:.4f}")
    print(f"Matched Treatment CR: {matched_treatment_cr:.4f}")
    print(f"Average Treatment Effect: {ate:.4f}")
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    # 전환율 비교 그래프
    plt.subplot(1, 2, 1)
    bars = plt.bar(['Control', 'Treatment'], 
                  [matched_control_cr, matched_treatment_cr],
                  color=['blue', 'orange'])
    plt.title('Conversion Rates After Matching')
    plt.ylabel('Conversion Rate')
    
    # 시간대별 전환율
    plt.subplot(1, 2, 2)
    hour_conversion = matched_df.groupby('most_ads_hour')['converted'].mean()
    hour_conversion.plot(kind='line', marker='o')
    plt.title('Conversion Rate by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Conversion Rate')
    
    plt.tight_layout()
    plt.show()
    
    return ate, {
        'original_control_cr': original_control_cr,
        'original_treatment_cr': original_treatment_cr,
        'matched_control_cr': matched_control_cr,
        'matched_treatment_cr': matched_treatment_cr,
        'ate': ate
    }

def main():
    # 분석 실행
    file_path = r"C:\Users\davie\Desktop\github\causal-inference-capstone-project\data\marketing_AB.csv"
    df = load_and_explore_data(file_path)
    processed_df = preprocess_data(df)
    matched_df = perform_psm(processed_df)
    logit_results, vif_data, importance = perform_logit_analysis(matched_df)
    ate, conversion_metrics = calculate_insights(df, matched_df)
    
    # 결과 저장 (같은 디렉토리에 저장)
    report_path = r"C:\Users\davie\Desktop\github\causal-inference-capstone-project\data\analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("=== 마케팅 A/B 테스트 분석 보고서 ===\n\n")
        f.write(f"데이터 크기: {len(df):,} rows\n")
        f.write(f"매칭 후 데이터 크기: {len(matched_df):,} rows\n\n")
        
        f.write("=== 전환율 분석 ===\n")
        f.write(f"원본 Control 전환율: {conversion_metrics['original_control_cr']:.4f}\n")
        f.write(f"원본 Treatment 전환율: {conversion_metrics['original_treatment_cr']:.4f}\n")
        f.write(f"매칭 후 Control 전환율: {conversion_metrics['matched_control_cr']:.4f}\n")
        f.write(f"매칭 후 Treatment 전환율: {conversion_metrics['matched_treatment_cr']:.4f}\n")
        f.write(f"평균 처치 효과(ATE): {ate:.4f}\n\n")
        
        f.write("=== 변수 중요도 ===\n")
        f.write(importance.to_string())
        f.write("\n\n=== VIF 분석 결과 ===\n")
        f.write(vif_data.to_string())
        f.write("\n\n=== Logit 분석 결과 ===\n")
        f.write(logit_results.summary().as_text())

if __name__ == "__main__":
    main() 