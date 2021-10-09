#from lightfm.cross_validation import random_train_test_split
#from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
#from lightfm import LightFM
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
item = pd.read_csv('items.csv').drop(['Unnamed: 0'], axis=1)
user = pd.read_csv('users.csv').drop(['Unnamed: 0'], axis=1)
print(len(user))
user_item = pd.read_csv('user_item.csv').drop(['Unnamed: 0'],axis=1)
user_item = user_item.sort_values(by = 'user_id')
user_item_us = user_item.user_id.values
tmp_user = user_item.user_id[0]
tmp_count = 0
users_dict = {}
for j in range(1, len(user_item)):
    if user_item_us[j] != tmp_user:
        users_dict[tmp_user] = [tmp_count, j]
        tmp_count = j
        tmp_user = user_item_us[j]

users_dict[tmp_user] = [tmp_count, len(user_item_us)]
user = user.sort_values(by = 'user_id')
tmp = user_item.user_id.values[len(user_item) // 10]
data = user_item[:users_dict[tmp][1]]
user_item_interaction = pd.pivot_table(data, index='user_id', columns='item_id', values='mark')
user_item_interaction = user_item_interaction.fillna(0)
user_id = list(user_item_interaction.index)
user_dict = {}
counter = 0
for i in user_id:
    user_dict[i] = counter
    counter += 1
user_item_interaction_csr = csr_matrix(user_item_interaction.values)
user_item_interaction_csr
model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)

model = model.fit(user_item_interaction_csr,
                  epochs=100,
                 verbose=False)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

