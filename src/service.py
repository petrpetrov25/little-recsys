import pandas as pd
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
import pyarrow.parquet as pq
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from lightfm.data import Dataset
import pickle
with open('/srv/model.pkl', 'rb') as f:
    model = pickle.load(f)
print('model loaded')
user_to_rec = input("user_id:")
table2 = pq.read_table('items.parquet')
item = table2.to_pandas()
table2 = pq.read_table('user_item.parquet')
user_item = table2.to_pandas()
table2 = pq.read_table('users.parquet')
user = table2.to_pandas()
user = user.sort_values(by='user_id')
user_item = user_item.sort_values(by='user_id')
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
tmp = user_item.user_id.values[len(user_item) // 40]
user_item = user_item[:users_dict[tmp][1]]
user_item_interaction = pd.pivot_table(user_item, index='user_id', columns='item_id', values='mark')
user_item_interaction = user_item_interaction.fillna(0)
user_item_interaction_csr = csr_matrix(user_item_interaction.values)


def _shuffle(uids, iids, data, random_state):
    shuffle_indices = np.arange(len(uids))
    random_state.shuffle(shuffle_indices)

    return (uids[shuffle_indices],
            iids[shuffle_indices],
            data[shuffle_indices])


def random_train_test_split(interactions_df,
                            test_percentage=0.25,
                            random_state=None):

    interactions = csr_matrix(interactions_df.values)
    if random_state is None:
        random_state = np.random.RandomState()

    interactions = interactions.tocoo()

    shape = interactions.shape
    uids, iids, data = (interactions.row,
                        interactions.col,
                        interactions.data)

    uids, iids, data = _shuffle(uids, iids, data, random_state)

    cutoff = int((1.0 - test_percentage) * len(uids))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = coo_matrix((data[train_idx],
                        (uids[train_idx],
                         iids[train_idx])),
                       shape=shape,
                       dtype=interactions.dtype)
    test = coo_matrix((data[test_idx],
                       (uids[test_idx],
                        iids[test_idx])),
                      shape=shape,
                      dtype=interactions.dtype)

    return train, test


train, test = random_train_test_split(user_item_interaction)


def feature_colon_value(my_list, col_names):
    """
    Takes as input a list and prepends the columns names to respective values in the list.
    For example: if my_list = [1,1,0,'del'],
    resultant output = ['f1:1', 'f2:1', 'f3:0', 'loc:del']

    """
    result = []
    ll = []
    for col in col_names:
        ll.append(col + ':')
    aa = my_list
    for x, y in zip(ll, aa):
        res = str(x) + "" + str(y)
        result.append(res)
    return result


def features_for_lightfm_dataset(col, df):
    ad_subset = df[col]
    ad_list = [list(x) for x in ad_subset.values]
    feature_list = []
    for x in ad_list:
        feature_list.append(feature_colon_value(x, col))
    features_result = []
    cls = []
    unique_ftrs = []
    for col in df.columns:
        cls += [col] * len(df[col].unique())
        unique_ftrs += list(df[col].unique())
    for x, y in zip(cls, unique_ftrs):
        res = str(x) + ":" + str(y)
        features_result.append(res)
    return features_result, feature_list


dataset1 = Dataset()
u_f, feature_list_user = features_for_lightfm_dataset(user.columns[1:], user[user.columns[1:]])
i_f, feature_list_item = features_for_lightfm_dataset(item.columns[1:], item[item.columns[1:]])
user_tuple = list(zip(user.user_id, feature_list_user))
item_tuple = list(zip(item.item_id, feature_list_item))
dataset1.fit(
    user['user_id'].unique(),  # all the users
    item['item_id'].unique(),  # all the items
    user_features=u_f,  # additional user features
    item_features=i_f  # additional item features
)
user_features = dataset1.build_user_features(user_tuple, normalize=False)
item_features = dataset1.build_item_features(item_tuple, normalize=False)


def sample_recommendation_user(model, interactions, user_features, item_features, user_id, user_dict,
                               users_dict, nrec_items=1, show=False):
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(
        model.predict(user_x, np.arange(n_items), user_features=user_features, item_features=item_features))
    scores.index = interactions.columns
    # print(scores.sort_values(ascending=False))
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    # print(type(scores[0]))

    known_items = user_item.item_id.values[users_dict[user_id][0]:users_dict[user_id][1]]
    # print(known_items)

    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]

    return return_score_list


user_id = list(user_item_interaction.index)
user_dict = {}
counter = 0
for i in user_id:
    user_dict[i] = counter
    counter += 1

print('recomended item is', sample_recommendation_user(model, user_item_interaction, user_features, item_features, user_to_rec, user_dict, users_dict, nrec_items=1))
