import pandas as pd
import numpy as np
import streamlit as st
from tqdm.auto import tqdm

def model_part1_gd(r: np.ndarray, alpha: int) -> np.ndarray:
    beta_user = np.random.random(r.shape[0])
    beta_item = np.random.random(r.shape[1])

    irow, jcol = np.where(~np.isnan(r))

    y = r.copy()

    for k in tqdm(range(1000)):

        y_pred = np.ones(r.shape) * np.nan

        for i, j in zip(irow, jcol):

            y_pred[i][j]= beta_user[i] + beta_item[j]

        g_beta_user = -1 * np.nansum(np.dstack((y, -y_pred)))
        g_beta_item = -1 * np.nansum(np.dstack((y, -y_pred)))

        beta_user_prev = beta_user.copy()
        beta_item_prev = beta_item.copy()

        beta_user = np.nansum(np.dstack((beta_user, - alpha * g_beta_user * np.ones(beta_user.shape)))[0], 1)  
        beta_item = np.nansum(np.dstack((beta_item, - alpha * g_beta_item * np.ones(beta_item.shape)))[0], 1)
        
        if np.linalg.norm(beta_user - beta_user_prev) < 0.00001:
            print(f"I do early stoping at iteration {k}")
            break

    return beta_user, beta_item

def model_part2_gd(r: np.ndarray, alpha: int, lambdar: float) -> np.ndarray:
    beta_user = np.random.random(r.shape[0])
    beta_item = np.random.random(r.shape[1])

    irow, jcol = np.where(~np.isnan(r))

    y = r.copy()

    for k in tqdm(range(100)):

        y_pred = np.ones(r.shape) * np.nan

        for i, j in zip(irow, jcol):

            y_pred[i][j]= beta_user[i] + beta_item[j]

        g_beta_user = -1 * np.nansum(np.dstack((y, -y_pred))) + lambdar * np.nansum(beta_user)
        g_beta_item = -1 * np.nansum(np.dstack((y, -y_pred))) + lambdar * np.nansum(beta_item)

        beta_user_prev = beta_user.copy()
        beta_item_prev = beta_item.copy()

        beta_user = np.nansum(np.dstack((beta_user, - alpha * g_beta_user * np.ones(beta_user.shape)))[0], 1)  
        beta_item = np.nansum(np.dstack((beta_item, - alpha * g_beta_item * np.ones(beta_item.shape)))[0], 1)
        
        if np.linalg.norm(beta_user - beta_user_prev) < 0.00001:
            print(f"I do early stoping at iteration {k}")
            break

    return beta_user, beta_item

if __name__ == '__main__':

    df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'])

    r = df.pivot(index='user_id', columns='item_id', values='rating').values

    #### PART 1

    r_copy = r.copy()

    irow, jcol = np.where(~np.isnan(r))

    idx = np.random.choice(np.arange(100_000), 1_000, replace=False)

    test_irow = irow[idx]
    test_jcol = jcol[idx]

    for (i, j) in zip(test_irow, test_jcol):
        r_copy[i][j] = np.nan

    beta_user, beta_item = model_part1_gd(r=r_copy, alpha=1e-7)

    err = []
    for i, j in zip(irow, jcol):
        y_pred = beta_user[i] + beta_item[j]
        y = r_copy[i][j]

        err.append((y_pred - y) ** 2)

    print(f"Part 1 Gradient Descent")
    print(f"RMSE in the test set: {np.sqrt(np.nanmean(np.array(err)))}")

    #### PART 2

    r_copy = r.copy()

    irow, jcol = np.where(~np.isnan(r))

    idx = np.random.choice(np.arange(100_000), 1_000, replace=False)
    test_irow = irow[idx]
    test_jcol = jcol[idx]

    for (i, j) in zip(test_irow, test_jcol):
        r_copy[i][j] = np.nan

    lambdar_range = np.array(list(range(20,110,20))) / 100

    rmse_list = []
    for lmbdr in lambdar_range:

        beta_user, beta_item = model_part2_gd(r=r_copy, alpha=1e-7, lambdar=lmbdr)

        err = []
        for i, j in zip(irow, jcol):
            y_pred = beta_user[i] + beta_item[j]
            y = r_copy[i][j]

            err.append((y_pred - y) ** 2)

        rmse = np.sqrt(np.nanmean(np.array(err)))
        rmse_list.append(rmse)

        # print(f"For Lambda {lmbdr} RMSE: {rmse}")
    print("Part 2 Gradient Descent with L2 regularization")
    print(f"Best Lambda value in the test set is {lambdar_range[np.argmin(rmse_list)]}")
    print("Best value selected by using grid search between 0.2 0.4 0.6 0.8 and 1.0")
    print(f"RMSE in the test set {rmse_list[np.argmin(rmse_list)]}")

            
            




