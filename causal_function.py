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
    df = raw[[x, y]].copy()
    tab = df.groupby(x).agg(['mean', 'sem']).round(2) # 평균, 표준오차
    tab.columns = ['mean', 'sem']
    tab['CI_lower'] = tab.apply(lambda x: x['mean'] - 2 * x['sem'], axis=1) # 95% 신뢰구간
    tab['CI_upper'] = tab.apply(lambda x: x['mean'] + 2 * x['sem'], axis=1)

    t_stat, p_value = stats.ttest_ind(df.loc[df[x] == tab.index[0], y], 
                                      df.loc[df[x] == tab.index[1], y], equal_var=True)
    
    res = {
            'pre_treat': tab.iloc[0]['mean'],
            'change_coef': round(tab.iloc[1]['mean'] - tab.iloc[0]['mean'], 2),
            'post_treat': tab.iloc[1]['mean'],
            'change_perc': round((tab.iloc[1]['mean'] - tab.iloc[0]['mean']) / tab.iloc[0]['mean'] * 100, 2),
            't_stat': t_stat.round(4),
            'p_value': p_value.round(4)
    }

    return res


# In[19]:


#ab_test(raw, 'Group', 'Time Spent')


def did (raw, x, y, start_dt, end_dt, treat_dt):
    
    # test
    res = {
            'pre_treat': 305141,
            'change_coef': round(8741.2, 2),
            'post_treat': 313286,
            'change_perc': round(8741.2/305141*100, 2),
            'p_value': 0.01
    }

    return res


def rd (raw, x, y, start_dt, end_dt, treat_dt):
    
    # test
    res = {
            'pre_treat': 310625,
            'change_coef': round(5884, 2),
            'post_treat': 299206,
            'change_perc': round(5884/310625*100, 2),
            'p_value': 0.01
    }

    return res


