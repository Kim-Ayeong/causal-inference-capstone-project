#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import datetime

import requests
from bs4 import BeautifulSoup
import time

#import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[17]:


def ab_test (raw, x, y):
    try:
        raw[y] = raw[y].map(lambda x: 1 if (str(x).lower() == 'true') or (str(x).lower() == 'yes') else 0)
    except:
        pass
    
    df = raw[[x, y]].copy()
    tab = df.groupby(x).agg(['mean', 'sem']).round(2) # 평균, 표준오차
    tab.columns = ['mean', 'sem']
    tab['CI_lower'] = tab.apply(lambda x: x['mean'] - 2 * x['sem'], axis=1) # 95% 신뢰구간
    tab['CI_upper'] = tab.apply(lambda x: x['mean'] + 2 * x['sem'], axis=1)

    t_stat, p_value = stats.ttest_ind(df.loc[df[x] == tab.index[0], y], 
                                      df.loc[df[x] == tab.index[1], y], equal_var=True)
    
    res = {
            'pre_treat': tab.iloc[0]['mean'],
            'post_treat': tab.iloc[1]['mean'],
            'change_coef': round(tab.iloc[1]['mean'] - tab.iloc[0]['mean'], 2),
            'change_perc': round((tab.iloc[1]['mean'] - tab.iloc[0]['mean']) / tab.iloc[0]['mean'] * 100, 2),
            't_stat': t_stat.round(4),
            'p_value': p_value.round(4)
    }

    return res


# In[19]:


#ab_test(raw, 'Group', 'Time Spent')


def did (raw, x, y, fix, treat_group, control_group, dt, start_dt, treat_dt, end_dt):
    try:
        raw[dt] = raw[dt].map(lambda x: x.date())
    except:
        pass
    df = raw.loc[(raw[dt] >= start_dt) & 
             (raw[dt] <= end_dt), [x, y, fix, dt]].copy()
    df['treated'] = df[x].map(lambda x: 1 if x == treat_group else 0)
    df['post'] = df[dt].map(lambda x: 1 if x >= treat_dt else 0)

    tab = pd.DataFrame(df.groupby(['treated', 'post'])[y].mean())
    model = smf.ols(f'{y} ~ treated:post + C({x}) + C({fix}) + C({dt})', data=df).fit()
    #model.summary()
    #model.conf_int().loc['treated:post'] # 95% 신뢰구간
    
    res = {
            'pre_treat': tab.loc[1, 0][y].round(1),
            'post_treat': tab.loc[1, 1][y].round(1),
            'change_value':tab.loc[1, 1][y].round(1) - tab.loc[1, 0][y].round(1),
            'change_coef': model.params['treated:post'].round(2),
            'change_perc': round(model.params['treated:post']/tab.loc[1, 0][y]*100, 2),
            'p_value': model.pvalues['treated:post'].round(2)
    }

    return res


def rd (raw, x, y, fix, treat_group, dt, start_dt, treat_dt, end_dt):
    try:
        raw[dt] = raw[dt].map(lambda x: x.date())
    except:
        pass
    df = raw.loc[(raw[dt] >= start_dt) & 
                 (raw[dt] <= end_dt) & 
                 (raw[x] == treat_group), [y, fix, dt]].copy()
    
    df['treated'] = (df[dt] >= treat_dt).astype(int)
    df['diff'] = (df[dt] - treat_dt).map(lambda x: x.days).astype(int)

    tab = pd.DataFrame(df.groupby('treated')[y].mean())
    
    model = smf.ols(f'{y} ~ treated:diff + C({fix})', data=df).fit()
    #model.summary()
    #model.conf_int().loc['treated:post'] # 95% 신뢰구간
    
    res = {
            'pre_treat': tab.loc[0][y].round(1),
            'post_treat': tab.loc[1][y].round(1),
            'change_value':tab.loc[1][y].round(1) - tab.loc[0][y].round(1),
            'change_coef': model.params['treated:diff'].round(2),
            'change_perc': round(model.params['treated:diff']/tab.loc[0][y]*100, 2),
            'p_value': model.pvalues['treated:diff'].round(2)
    }

    return res


