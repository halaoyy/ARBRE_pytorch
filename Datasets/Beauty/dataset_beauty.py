# coding: utf-8
import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

if __name__ == "__main__":
    # Load raw data
    print("---------------------Raw data loading-----------------------")
    f = open('Beauty.txt', 'r')
    user, item, clk = [], [], []
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        user.append(u)
        item.append(i)
        clk.append(int(1))

    sample_dict = {
        'user_id': user,
        'item_id': item,
        'clk': clk
    }
    sample = pd.DataFrame(sample_dict)

    if not os.path.exists('data_beauty/'):
        os.mkdir('data_beauty/')

    if not os.path.exists('data_beauty/Beauty_info.pkl'):
        sample.columns = ['user_id', 'item_id', 'clk']
        print("Original sample num: ", sample.shape[0])

        # Data cleaning
        print("---------------------Data cleaning--------------------------")
        clean = 0
        while (clean == 0):
            clean = 1
            sample_user = sample['user_id']
            user_count = Counter(sample_user)
            user_valid = []
            for u in user_count.keys():
                if user_count[u] >= 5:
                    user_valid.append(u)

            sample_item = sample['item_id']
            item_count = Counter(sample_item)
            item_valid = []
            for i in item_count.keys():
                if item_count[i] >= 5:
                    item_valid.append(i)

            sample = sample[sample['user_id'].isin(user_valid)]
            sample = sample[sample['item_id'].isin(item_valid)]

            sample_user = sample['user_id']
            user_count = Counter(sample_user)
            for u in user_count.keys():
                if user_count[u] < 5:
                    clean = 0
                    break

            sample_item = sample['item_id']
            item_count = Counter(sample_item)
            for i in item_count.keys():
                if item_count[i] < 5:
                    clean = 0
                    break

        print("Numbers of samples after cleaning: ", sample.shape[0])
        user_set = sample['user_id'].unique()
        user_set = pd.DataFrame(user_set, columns=['user_id'])
        user_set = user_set.sample(frac=1, random_state=1234)

        sample = sample.loc[sample.user_id.isin(user_set.user_id.unique())]

        print("---------------------Dataset statistics---------------------")
        # encode user_id & item_id
        lbe_user = LabelEncoder()
        unique_user_id = sample['user_id'].unique()
        lbe_user.fit(unique_user_id)
        sample['user_id'] = lbe_user.transform(sample['user_id']) + 1

        lbe_item = LabelEncoder()
        unique_item_id = sample['item_id'].unique()
        lbe_item.fit(unique_item_id)
        sample['item_id'] = lbe_item.transform(sample['item_id']) + 1

        # get user_set & item_set
        user_set = sample['user_id'].unique()
        user_set = pd.DataFrame(user_set, columns=['user_id'])
        user_set.sort_values('user_id', inplace=True, ascending=True)

        num_users = user_set.shape[0]
        print("Numbers of users: ", num_users)

        item_set = sample['item_id'].unique()
        item_set = pd.DataFrame(item_set, columns=['item_id'])
        item_set.sort_values('item_id', inplace=True, ascending=True)
        num_items = item_set.shape[0]
        print("Numbers of items: ", num_items)

        user_set = user_set.values.tolist()
        item_set = item_set.values.tolist()

        # get sample_set
        sample['clk'] = sample.clk.apply(lambda x: 1)
        num_samples = sample.shape[0]
        print("Numbers of samples: ", num_samples)

        # get density
        density = num_samples / (num_users * num_items)
        print("Density: ", density)

        sample_set = sample.values.tolist()
        sample.sort_values('user_id', inplace=True, ascending=True)

        with open('data_beauty/Beauty_info.pkl', 'wb') as f:
            pickle.dump(user_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(item_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(sample_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((num_users, num_items), f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Dataset information already generated.")
