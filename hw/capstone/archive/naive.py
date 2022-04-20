# %%
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import datetime

transactions = pd.read_csv('data/transactions.csv', parse_dates=['t_dat'])
articles = pd.read_csv('data/articles.csv')
customers = pd.read_csv('data/customers.csv')

# %%
start_train = datetime.date(2020, 6, 22)
start_test = datetime.date(2020, 9, 15)

tran_test = transactions[transactions.t_dat > pd.Timestamp(start_test)]
tran_train = transactions[(transactions.t_dat <= pd.Timestamp(start_test))& (transactions.t_dat > pd.Timestamp(start_train))] # 

del transactions

# %%
tran_test_feat = tran_test.merge(customers, on='customer_id').merge(articles, on='article_id').dropna()

# %%
tran_train_feat = tran_train.merge(customers, on='customer_id').merge(articles, on='article_id').dropna()

# %%
object_columns = ['sales_channel_id', 'club_member_status', 'fashion_news_frequency',
'product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id',# 'product_code',
'perceived_colour_master_id', 'department_no', 'index_group_no', 'section_no', 'garment_group_no']
non_object_columns = ['age']

cols_to_use = non_object_columns + object_columns

y_train = tran_train_feat['price']
X_train = tran_train_feat[cols_to_use]
y_test = tran_test_feat['price']
X_test = tran_test_feat[cols_to_use]

# %%
from sklearn.preprocessing import OneHotEncoder
X_train = X_train.astype(object)
X_train['age'] = X_train['age'].astype(int)
X_train['age'] = (X_train['age'] - X_train['age'].min()) / (X_train['age'].max()- X_train['age'].min())

X_test = X_test.astype(object)
X_test['age'] = X_test['age'].astype(int)
X_test['age'] = (X_test['age'] - X_test['age'].min()) / (X_test['age'].max()- X_test['age'].min())

X = X_train.append(X_test)

encoder = OneHotEncoder()
encoder.fit(X[object_columns])

del X

# %%
from sklearn.linear_model import SGDRegressor
model = SGDRegressor()

# %%
batch_size = 1024
for start_ind in tqdm(range(0, X_train.shape[0], batch_size)):
    end_ind = start_ind + batch_size
    X_train_part = X_train.iloc[start_ind:end_ind]
    y_train_part = y_train.iloc[start_ind:end_ind]

    X_train_part_encoded = encoder.transform(X_train_part[object_columns]).todense()

    train_categorized = np.concatenate((X_train_part[non_object_columns].values, X_train_part_encoded), axis=1)

    model.partial_fit(train_categorized, y_train_part)

# %%
batch_size = 1024

preds = np.array([])
labels = np.array([])

for start_ind in tqdm(range(0, X_test.shape[0], batch_size)):
    end_ind = start_ind + batch_size
    X_test_part = X_test.iloc[start_ind:end_ind]
    y_test_part = y_test.iloc[start_ind:end_ind]

    X_test_part_part_encoded = encoder.transform(X_test_part[object_columns]).todense()

    test_categorized = np.concatenate((X_test_part[non_object_columns].values, X_test_part_part_encoded), axis=1)
    pred = model.predict(test_categorized)

    label = y_test.iloc[start_ind:end_ind].values

    preds = np.concatenate((preds, pred), axis=0)
    labels = np.concatenate((labels, label), axis=0)


# %%
errs = preds - labels
rmse = np.sqrt(np.mean(errs ** 2))
mae = np.mean(np.abs(errs))

print(f"Score in test set mae: {mae:.3f}, rmse: {rmse:.3f}")


