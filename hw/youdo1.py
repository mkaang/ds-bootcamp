import numpy as np
import pandas as pd
import streamlit as st
import plotly_express as px
from sklearn.datasets import fetch_california_housing
import plotly.graph_objects as go
import wedo1

def main(verbosity=False):
    st.header("Build a custom regression model that meet the business demand")
    st.markdown("""
    Tasks
    - Define a custom loss function for a given threshold that defines the error tolerance of a single instance
    - Plot the custom loss function to see whether it is conxev or not
    - Write down all the loss function and gradient equations in Latex
    - Find the Beta1 and Beta0
    - Add L2 regularization to the loss function
    - Compare the fitted regression line with the one in the wedo.
    """)

    st.markdown("#### Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    st.dataframe(X)
    
    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target, HouseAgeGroup=(X['HouseAge'].values / 10).astype(np.int)))

    st.dataframe(df)

    st.markdown("#### Model")
    st.latex(
        r"\begin{equation}" + 
        r"\hat{y}_i=\beta_0 + \beta_1 x_i" +
        r"\end{equation}"
    )

    st.markdown("#### Loss Function")
    st.latex(
        r"\begin{equation}" + 
        r"L(\beta_0,\beta_1)=\sum_{i=1}^{N}{(\max(\theta, |y_i - \hat{y}_i|))^2} + \lambda (\beta_0^2 + \beta_1^2), \ \ \ \ \  \lambda > 0" +
        r"\end{equation}"
    )
    st.latex(
        r"\begin{equation}" + 
        r"L(\beta_0,\beta_1)=" + 
        r"\begin{cases}" + 
        r"\sum_{i=1}^{N}{\theta^2} + \lambda (\beta_0^2 + \beta_1^2), & \text{if}\ |y_i - \hat{y}_i| <= \theta , \ \ \ \ \  \lambda > 0 \\" + 
        r"\sum_{i=1}^{N}{(y_i - \hat{y}_i)^2} + \lambda (\beta_0^2 + \beta_1^2), & \text{otherwise} , \ \ \ \ \  \lambda > 0" + 
        r"\end{cases}" + 
        r"\end{equation}"
        )


    st.markdown("#### Partial derivatives")
    st.latex(
        r"\begin{equation}" + 
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_0}=" + 
        r"\begin{cases}" + 
        r"2 \lambda \beta_0, & \text{if}\ |y_i - \hat{y}_i| <= \theta , \ \ \ \ \  \lambda > 0 \\" + 
        r"-2\sum_{i=1}^{N}{(y_i - \hat{y}_i)} + 2 \lambda \beta_0, & \text{otherwise} , \ \ \ \ \  \lambda > 0 " + 
        r"\end{cases}" + 
        r"\end{equation}"
    )
    st.latex(
        r"\begin{equation}" + 
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_0}=" + 
        r"\begin{cases}" + 
        r"2 \lambda \beta_1, & \text{if}\ |y_i - \hat{y}_i| <= \theta , \ \ \ \ \  \lambda > 0 \\" + 
        r"-2\sum_{i=1}^{N}{(y_i - \hat{y}_i)x_i } + 2 \lambda \beta_1, & \text{otherwise} , \ \ \ \ \  \lambda > 0 " + 
        r"\end{cases}" + 
        r"\end{equation}"
    )
    
    st.markdown("#### Is the loss function really convex? ")
    st.markdown("Check this out and see how it is look like")

    theta = st.slider("Threshold", 0.0, 10.0, value=0.1)
    lambdar = st.slider("Lambda", 0.0, 1.0, value=0.1)

    losses = list()
    for beta in np.linspace(-1000, 1000, 100) / 100:
        loss = (y - beta * X['MedInc']) - 2 * lambdar * beta 
        loss = np.where(np.abs(loss)<theta, theta, loss)
        losses.append(np.sum(np.power(loss, 2)))

    l = pd.DataFrame(dict(beta1=np.linspace(-1000, 1000, 100) / 100, losses=losses))

    fig = px.scatter(l, x="beta1", y="losses")
    st.plotly_chart(fig, use_container_width=True)

    losses = list()
    for beta in np.linspace(-1000, 1000, 100) / 100:
        loss = (y - beta) - 2 * lambdar * beta 
        loss = np.where(np.abs(loss)<theta, theta, loss)
        losses.append(np.sum(np.power(loss, 2)))

    l = pd.DataFrame(dict(beta0=np.linspace(-1000, 1000, 100) / 100, losses=losses))

    fig = px.scatter(l, x="beta0", y="losses")
    st.plotly_chart(fig, use_container_width=True)

    if verbosity: 
        st.dataframe(l)

    st.markdown(f"#### YouDo Model with threshold={theta}, lambda={lambdar} hyper-parameter selection")
    beta_yd = reg(df['MedInc'].values, df['Price'].values, theta=theta, lambdar=lambdar, verbose=verbosity)
    st.latex(fr"Price = {beta_yd[1]:.4f} \times MedInc + {beta_yd[0]:.4f}")


    p = st.slider("Mixture Ratio (p)", 0.0, 1.0, value=0.8)
    st.markdown(f"#### WeDo model with p={p:.2f} contribution")
    beta_wd, gamma = wedo1.reg(df['MedInc'].values, df['Price'].values, df['HouseAgeGroup'].values,
                      p=p,
                      verbose=verbosity)
    st.latex(fr"Price = {gamma[0][1]:.4f} \times MedInc + {gamma[0][0]:.4f}")
    st.latex(fr"Price = {gamma[1][1]:.4f} \times MedInc + {gamma[1][0]:.4f}")
    st.latex(fr"Price = {gamma[2][1]:.4f} \times MedInc + {gamma[2][0]:.4f}")
    st.latex(fr"Price = {gamma[3][1]:.4f} \times MedInc + {gamma[3][0]:.4f}")
    st.latex(fr"Price = {gamma[4][1]:.4f} \times MedInc + {gamma[4][0]:.4f}")
    st.latex(fr"Price = {gamma[5][1]:.4f} \times MedInc + {gamma[5][0]:.4f}")


    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df['MedInc'].values, y=y, mode='markers', name='data points'))

    y_pred_yd = beta_yd[0] + beta_yd[1] * df['MedInc'].values
    fig.add_trace(
            go.Scatter(x=df['MedInc'].values, y=y_pred_yd, mode='lines', name='youdo'))

    y_pred_wd = beta_wd[0] + beta_wd[1] * df['MedInc'].values
    fig.add_trace(
            go.Scatter(x=df['MedInc'].values, y=y_pred_wd, mode='lines', name='wedo'))

    st.plotly_chart(fig, use_container_width=True)

def reg(x, y, theta, lambdar, verbose=False):
    beta = np.random.random(2)

    if verbose:
        st.write(beta)
        st.write(x)

    alpha = 1e-4
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _x, _y in zip(x, y):
            y_pred = beta[0] + beta[1] * _x

            if np.abs(_y - y_pred) < theta:
                g_b0 = 0
                g_b1 = 0
            else:
                # print(f"Beta: {beta}")
                g_b0 = -2 * (_y - y_pred) + 2 * lambdar * beta[0]
                g_b1 = -2 * ((_y - y_pred) * _x) + 2 * lambdar * beta[1]


            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            # beta - err / g_b0 ? newton hampthon

            err += max([theta, (_y - y_pred)]) ** 2

        print(f"{it} - Beta: {beta}, Error: {err}")
        my_bar.progress(it / n_max_iter)

    return beta

if __name__ == '__main__':
    main(st.sidebar.checkbox("verbosity"))