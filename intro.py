import numpy as np
import streamlit as st
from sklearn.datasets import make_regression
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from typing import Callable
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score


def mode(x: np.ndarray) -> int:
    return Counter(x).most_common(1)[0][0]


def weighted_mode(clazzes: np.ndarray, weight: np.ndarray) -> int:
    ws = {}
    for c, w in zip(clazzes, weight):
        ws[c] = ws.get(c, 0) + w

    return max(ws.items(), key=lambda t: t[1])[0]


def knn(x: np.ndarray, y: np.ndarray, xspace: np.ndarray, k: int, dist_fn: Callable[[np.ndarray, np.ndarray], float],
        verbose: bool = True) -> np.ndarray:
    """


    :param x: Feature matrix
    :param y: Class of instance
    :param xspace: Data grid of space
    :param k: Number of neighbour
    :param dist_fn: Distance calculation function
    :param verbose: verbosity
    :return: Predicted class of each instance
    """
    m1 = xspace.shape[0]
    st.write(m1)
    m2 = x.shape[0]
    dist = np.empty((m1, m2))

    for i in range(m1):
        for j in range(m2):
            dist[i][j] = dist_fn(xspace[i, :], x[j, :])

    if verbose:
        st.write(dist)

    y_pred = []

    for i in range(m1):
        idx = np.argsort(dist[i, :])

        y_pred.append(mode(y[idx[:k]]))

    return y_pred


# Callable[[np.ndarray, np.ndarray], float]
def euc(x1: np.ndarray, x2: np.ndarray) -> float:
    return sqrt(((x1 - x2) ** 2).sum())


# l2
# euclidean
# norm-2
def euc2(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.linalg.norm(x1 - x2)


def l1(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(np.abs(x1 - x2))


def main(verbose: bool = True):
    st.header("Let's Generate a Blob")

    rng = np.random.RandomState(0)
    x, y = make_regression(10, 1, random_state=rng, noise=st.slider("Noise", 1., 100., value=10.))

    if st.checkbox("Outlier"):
        x = np.concatenate((x, np.array([[-2], [0], [2], [-2], [0], [2]])))
        y = np.concatenate((y, np.array([300, 300, 300, 300, 300, 300])))

    df = pd.DataFrame(dict(x=x[:, 0], y=y))

    if verbose:
        st.dataframe(df)

    fig = px.scatter(df, x="x", y="y")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model 1")
    st.latex("y = a")
    st.write("""Find the **optimal** that will minimize my **error**
    """)

    if verbose:

        loss0, loss, loss2, loss3 = [], [], [], []

        for a in np.linspace(-100, 100, 100):
            loss0.append(np.abs(y - a).sum())
            loss.append(np.power((y - a), 2).sum())
            loss2.append(np.power((y - a), 2).mean())
            loss3.append(np.power((y - a), 4).mean())

        l = pd.DataFrame(dict(a=np.linspace(-100, 100, 100), loss0=loss0, loss=loss, loss2=loss2, loss3=loss3))

        st.dataframe(l)

        fig = px.scatter(l, x="a", y="loss")

        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(l, x="a", y="loss2")

        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(l, x="a", y="loss3")

        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(l, x="a", y="loss0")

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model 2")
    st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")
    st.write("""Find the **optimal** betas that will minimize my **error**
    """)

    st.latex(r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 }")

    if verbose:
        loss, b0, b1 = [], [], []
        for i, _b0 in enumerate(np.linspace(-100, 100, 50)):
            if i == 30:
                for _b1 in np.linspace(-100, 100, 50):
                    b0.append(_b0)
                    b1.append(_b1)

                    loss.append(np.power((y - _b1 * x - _b0), 2).sum())

        l = pd.DataFrame(dict(b0=b0, b1=b1, loss=loss))

        fig = px.scatter(l, x="b1", y="loss")

        st.plotly_chart(fig, use_container_width=True)

    beta1 = model1(x[:, 0], y, verbose=verbose)
    st.write(beta1)
    st.write(x)
    st.write(y)

    st.subheader("Model 3 (L2 Regularized)")
    st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")
    st.write("""Find the **optimal** betas that will minimize my **error**
    """)

    st.latex(
        r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (\beta_0^2 + \beta_1^2)")

    lam1 = st.slider("Regularization Multiplier for L2 (lambda)", 0.001, 10., value=0.1)
    beta2 = model2(x[:, 0], y, lam1)

    st.subheader("Model 3 (L1 Regularized)")
    st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")
    st.write("""Find the **optimal** betas that will minimize my **error**
        """)

    st.latex(
        r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (|\beta_0| + |\beta_1|)")

    lam2 = st.slider("Regularization Multiplier for L1(lambda)", 0.001, 10., value=0.1)
    beta3 = model3(x[:, 0], y, lam2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[:, 0], y=y, mode='markers', name='data points'))
    fig.add_trace(go.Scatter(x=x[:, 0], y=beta1[0] + beta1[1] * x[:, 0], mode='lines', name='regresssion'))
    fig.add_trace(go.Scatter(x=x[:, 0], y=beta2[0] + beta2[1] * x[:, 0], mode='lines', name='regression + L2'))
    fig.add_trace(go.Scatter(x=x[:, 0], y=beta3[0] + beta3[1] * x[:, 0], mode='lines', name='regression + L1'))
    # fig = px.scatter(df, x="x", y="y")

    st.plotly_chart(fig, use_container_width=True)

    loss, b0, b1 = [], [], []
    for i, _b0 in enumerate(np.linspace(-100, 100, 50)):
        for _b1 in np.linspace(-100, 100, 50):
            b0.append(_b0)
            b1.append(_b1)

            loss.append(np.power((y - _b1 * x - _b0), 2).sum() + lam1 * (_b0 ** 2 + _b1 ** 2))

    regloss = pd.DataFrame(dict(loss=loss, b0=b0, b1=b1))
    fig = px.density_contour(regloss, x="b0", y="b1", z="loss", histfunc="avg")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Polynomial dependencies with LR")

    rng = np.random.RandomState(0)
    x = np.random.random(100) * 32 - 16
    y = np.power(0.5 * x - 4, 4) + np.random.normal(scale=100., size=100)
    df = pd.DataFrame(dict(x=x, y=y))

    if verbose:
        st.dataframe(df)

    fig = px.scatter(df, x="x", y="y")

    st.plotly_chart(fig, use_container_width=True)

    st.latex(r"y = \beta_1 x + \beta_0")
    st.latex(r"y = \beta_1 x^4 + \beta_0")
    st.latex(r"z = x^4, y = \beta_1 z + \beta_0")

    st.latex(r"""
    z_1 = x \\
    z_2 = x^2  \\
    z_3 = x^3 \\
    z_4 = x^4 \\
    
     y = \beta_4 z_4  +  \beta_3 z_3 + \beta_2 z_2 + \beta_1 z_1  +\beta_0\\
     
     y = \beta^T z \\
     
     z = [1, x,  x^2, x^3, x^4 ]\\
     \beta = [\beta_0,\beta_1,\beta_2,\beta_3,\beta_4 ]
    
    """)

def model3(x, y, lam, alpha=0.0001) -> np.ndarray:
    # print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        if beta[0] >= 0:
            g_b0 = -2 * (y - y_pred).sum() + lam
        else:
            g_b0 = -2 * (y - y_pred).sum() - lam

        if beta[1] >= 0:
            g_b1 = -2 * (x * (y - y_pred)).sum() + lam
        else:
            g_b1 = -2 * (x * (y - y_pred)).sum() - lam

        # print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        # if np.linalg.norm(beta - beta_prev) < 0.000001:
        #     print(f"I do early stoping at iteration {i}")
        #     break

    return beta

def model2(x, y, lam, alpha=0.0001) -> np.ndarray:
    # print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        # print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        # if np.linalg.norm(beta - beta_prev) < 0.000001:
        #     print(f"I do early stoping at iteration {i}")
        #     break

    return beta

def model1(x, y, alpha=0.001, verbose=False) -> np.ndarray:
    beta = np.random.random(2)
    if verbose:
        st.write(beta)

    # print("starting sgd")
    for i in range(100):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum()
        g_b1 = -2 * (x * (y - y_pred)).sum()

        # print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        # if np.linalg.norm(beta - beta_prev) < 0.001:
        #     print(f"I do early stoping at iteration {i}")
        #     break

    return beta


def model(group , x, y, lam, alpha=0.001, verbose=False) -> np.ndarray:
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        # print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        # if np.linalg.norm(beta - beta_prev) < 0.000001:
        #     print(f"I do early stoping at iteration {i}")
        #     break

    return beta

if __name__ == '__main__':
    main(verbose=st.sidebar.checkbox("Verbose"))
