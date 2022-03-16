import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing

import intro

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

df = pd.DataFrame(dict(MedInc=X['MedInc'], Price=cal_housing.target, HouseAgeGroup=(X['HouseAge'].values / 10).astype(np.int)))

fig = px.scatter(df, x="MedInc", y="Price", color="HouseAgeGroup")

# beta1 = intro.model1(x=df['MedInc'], y=df["Price"], alpha=1e-6)
# beta2 = intro.model2(x=df['MedInc'], y=df["Price"], lam=0.5, alpha=1e-6)
# beta3 = intro.model3(x=df['MedInc'], y=df["Price"], lam=0.5, alpha=1e-6)
# print(beta1, beta2, beta3)

# _X = X[['MedInc']].copy()
# _X['bias'] = 1

# y_pred = (_X * beta1).sum(axis=1)
# error1 = np.mean(np.sqrt((y_pred - y)**2))

# y_pred = (_X * beta2).sum(axis=1)
# error2 = np.mean(np.sqrt((y_pred - y)**2))

# y_pred = (_X * beta3).sum(axis=1)
# error3 = np.mean(np.sqrt((y_pred - y)**2))

# print(error1, error2, error3)

beta = intro.model(group=df['HouseAgeGroup'], x=df['MedInc'], y=df["Price"], lam=0.5, alpha=1e-6)