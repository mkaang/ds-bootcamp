import streamlit as st
from sklearn.datasets import make_blobs
import pandas as pd
import plotly.express as px
import numpy as np

from typing import Callable
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score

def mode(x: np.ndarray) -> int:
    return Counter(x).most_common(1)[0][0]

def weighted_mode(x: np.ndarray, w: np.ndarray) -> int:
    return Counter(x).most_common(1)[0][0]

def algo(x: np.ndarray, y: np.ndarray, k: int, dist_fn:Callable[[np.ndarray, np.ndarray], float], verbose:bool = True) -> np.ndarray:

    m = x.shape[0]
    dist = np.empty((m, m))

    for i in range(m):
        for j in range(m):
            if i < j:
                dist[i][j] = dist_fn(x[i, :], x[j, :])
            elif i > j:
                dist[i][j] = dist[j][i] 
            else:
                dist[i][j] = 0

    if verbose:
        st.write(dist)

    y_pred = []

    for i in range(m):
        idx = np.argsort(dist[i, :])

        # y_pred.append(mode(y[idx[1:k+1]]))   
        y_pred.append(weighted_mode(y[idx[1:k+1]], dist[i, 1:k+1]))   

    return y_pred


# Callable[[np.ndarray, np.ndarray], float]
def euc(x1: np.ndarray, x2: np.ndarray) -> float:
    return sqrt(((x1-x2)**2).sum())

def euc2(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.linalg.norm(x1 - x2)

def l1(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(np.abs(x1-x2))

def main(verbose: bool = True):
    st.header("Lets generate a blob")

    m = st.slider("Nunmber of samples", 10, 1_000,value=100)
    sd = st.slider("Noise", 0., 10., value=.1)
    x, y = make_blobs(n_samples=m, centers=5, n_features=2, random_state=42, cluster_std=sd)

    df = pd.DataFrame(dict(x1=x[:,0], x2=x[:,1], y=y))

    if verbose:
        st.dataframe(df)

    fig = px.scatter(df, x="x1", y="x2", color="y")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Algorithm")

    st.markdown("""
    * A new instance is similar to *k* **nearest** neighbour
    * Strategies
        * Majority voting
        * **Weighted** Voting
    """)

    y_pred = algo(x, y, k=st.slider("Neighbour Count", 1, 5), dist_fn=l1, verbose=verbose)

    if verbose:
        st.write(y_pred)
        st.write(y.tolist())

    # acc = sum(1 for p,t in zip(y_pred, y) if p==t) / len(y_pred)
    # acc = accuracy_score(y, y_pred)

    acc = []
    for _k in range(1,11):
        y_pred = algo(x, y, k=_k, dist_fn=euc2, verbose=verbose)

        acc.append(accuracy_score(y, y_pred))

    accuracy = pd.DataFrame(dict(k=list(range(1,11)), accuracy=acc))

    st.write(accuracy)

if __name__ == '__main__':
    main(verbose=st.checkbox("Verbose", value=True))