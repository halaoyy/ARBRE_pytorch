# coding: utf-8
proj_path = "/home/data/zjz/oyy/ARBRE_pytorch/Datasets/Beauty/"  # change to your path
import sys
sys.path.append(proj_path)

import os
import random
import pickle
import numpy as np
import pandas as pd
import tqdm

random.seed(1234)

from config import *

if __name__ == '__main__':

    # Part-1 get user & item & sample infos
    with open('data_'+DATASETL+'/'+DATASETU+'_info.pkl', 'rb') as f:
        user_set = pickle.load(f)
        item_set = pickle.load(f)
        sample_set = pickle.load(f)
        (num_users, num_items) = pickle.load(f)
        print("dataset infos loaded")

    # Part-2 negative sample generation
    if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_val_negative_'+str(NEG_NUM)+'.pkl'):
        print("Generating negative samples for validation: ")
        sample_clk = pd.DataFrame(sample_set, columns=['user_id', 'item_id', 'clk'])
        sample_dict = pd.DataFrame(sample_clk, columns=['user_id', 'item_id']).groupby('user_id').apply(list).to_dict()
        neg_dict = dict()
        for u in tqdm.tqdm(sample_dict.keys()):
            neg_item = []
            for _ in range(NEG_NUM):
                i = np.random.randint(1, num_items+1)
                while i in sample_dict[u]:
                    i = np.random.randint(1, num_items+1)
                neg_item.append(i)
            neg_dict[u] = neg_item

        with open('data_'+DATASETL+'/'+DATASETU+'_val_negative_'+str(NEG_NUM)+'.pkl', 'wb') as f:
            pickle.dump(neg_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data_'+DATASETL+'/'+DATASETU+'_val_negative_'+str(NEG_NUM)+'.pkl', 'rb') as f:
            neg_dict = pickle.load(f)
            neg_dict_val = neg_dict
        print(DATASETU, "_val_negative_", NEG_NUM, "loaded")

    if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_eval_negative_'+str(NEG_NUM)+'.pkl'):
        print("Generating negative samples for test: ")
        sample_clk = pd.DataFrame(sample_set, columns=['user_id', 'item_id', 'clk'])
        sample_dict = pd.DataFrame(sample_clk, columns=['user_id', 'item_id']).groupby('user_id').apply(list).to_dict()
        neg_dict = dict()
        for u in tqdm.tqdm(sample_dict.keys()):
            neg_item = []
            for _ in range(NEG_NUM):
                i = np.random.randint(1, num_items+1)
                while i in sample_dict[u]:
                    i = np.random.randint(1, num_items+1)
                neg_item.append(i)
            neg_dict[u] = neg_item
        with open('data_'+DATASETL+'/'+DATASETU+'_eval_negative_'+str(NEG_NUM)+'.pkl', 'wb') as f:
            pickle.dump(neg_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data_'+DATASETL+'/'+DATASETU+'_eval_negative_'+str(NEG_NUM)+'.pkl', 'rb') as f:
            neg_dict = pickle.load(f)
            neg_dict_eval = neg_dict
        print(DATASETU, "_eval_negative_", NEG_NUM, "loaded")

    # Partie-3 get train & validation & test pos resources
    if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_train_val_test.pkl'):
        print("Generating train & validation & test data: ")
        sample_set = pd.DataFrame(sample_set, columns=['user_id', 'item_id', 'clk'])
        sample_set = sample_set.values.tolist()
        random.shuffle(sample_set)
        train_set = sample_set[:int(len(sample_set)*0.8)]
        val_set = sample_set[int(len(sample_set)*0.8):int(len(sample_set)*0.9)]
        test_set = sample_set[int(len(sample_set)*0.9):len(sample_set)]

        train_df = pd.DataFrame(train_set, columns=['user_id', 'item_id', 'clk'])
        val_df = pd.DataFrame(val_set, columns=['user_id', 'item_id', 'clk'])
        test_df = pd.DataFrame(test_set, columns=['user_id', 'item_id', 'clk'])

        train_pos_df = train_df[train_df['clk'] == 1]
        val_pos_df = val_df[val_df['clk'] == 1]
        test_pos_df = test_df[test_df['clk'] == 1]

        with open('data_'+DATASETL+'/'+DATASETU+'_train_val_test.pkl', 'wb') as f:
            pickle.dump(train_pos_df, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(val_pos_df, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_pos_df, f, pickle.HIGHEST_PROTOCOL)
            print("Done")
    else:
        with open('data_'+DATASETL+'/'+DATASETU+'_train_val_test.pkl', 'rb') as f:
            train_pos_df = pickle.load(f)
            val_pos_df = pickle.load(f)
            test_pos_df = pickle.load(f)

        print("train & val & test resources loaded")

    # Part-4 : Create user-item bipartite graph: get user_clk_dict, item_clk_dict
    if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_ui_clk.pkl'):
        print("Generating interaction data: ")
        user_clk_dict = train_pos_df.groupby('user_id')['item_id'].apply(list).to_dict()
        item_clk_dict = train_pos_df.groupby('item_id')['user_id'].apply(list).to_dict()

        with open('data_'+DATASETL+'/'+DATASETU+'_ui_clk.pkl', 'wb') as f:
            pickle.dump(user_clk_dict, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(item_clk_dict, f, pickle.HIGHEST_PROTOCOL)
            print("Done")
    else:
        with open('data_'+DATASETL+'/'+DATASETU+'_ui_clk.pkl', 'rb') as f:
            user_clk_dict = pickle.load(f)
            item_clk_dict = pickle.load(f)

        print(DATASETU, "_ui_clk loaded")

    # Part-5 : Create user-to-user / item-to-item collaborative graph: get neighborhood users / items
    if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_uu_sim_'+str(U_SIM_THRESHOLD)+'.pkl'):
        print("Generating user collaborative similar neighbors: ")
        user_sim = []
        for u in tqdm.tqdm(range(1, num_users+1)):
            if u in user_clk_dict.keys():
                u_clk = user_clk_dict[u]
                for v in range(u+1, num_users+1):
                    if v in user_clk_dict.keys():
                        v_clk = user_clk_dict[v]
                        num_common = len(set(u_clk) & set(v_clk))
                        num_base = len(set(u_clk) | set(v_clk))
                        if num_common / num_base >= U_SIM_THRESHOLD:
                            user_sim.append([u, v])
                            user_sim.append([v, u])

        user_sim_df = pd.DataFrame(user_sim, columns=['user1', 'user2'])

        with open('data_'+DATASETL+'/'+DATASETU+'_uu_sim_'+str(U_SIM_THRESHOLD)+'.pkl', 'wb') as f:
            pickle.dump(user_sim_df, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data_'+DATASETL+'/'+DATASETU+'_uu_sim_'+str(U_SIM_THRESHOLD)+'.pkl', 'rb') as f:
            user_sim_df = pickle.load(f)
        print(DATASETU, "_uu_sim_", str(U_SIM_THRESHOLD), " loaded")

    # item to item graph neighborhood
    if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_ii_sim_'+str(I_SIM_THRESHOLD)+'.pkl'):
        print("Generating item collaborative similar neighbors: ")
        item_sim = []
        for i in tqdm.tqdm(range(1, num_items+1)):
            if i in item_clk_dict.keys():
                i_clk = item_clk_dict[i]
                for j in range(i+1, num_items+1):
                    if j in item_clk_dict.keys():
                        j_clk = item_clk_dict[j]
                        num_common = len(set(i_clk) & set(j_clk))
                        num_base = len(set(i_clk) | set(j_clk))
                        if num_common / num_base >= I_SIM_THRESHOLD:
                            item_sim.append([i, j])
                            item_sim.append([j, i])

        item_sim_df = pd.DataFrame(item_sim, columns=['item1', 'item2'])

        with open('data_'+DATASETL+'/'+DATASETU+'_ii_sim_'+str(I_SIM_THRESHOLD)+'.pkl', 'wb') as f:
            pickle.dump(item_sim_df, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data_'+DATASETL+'/'+DATASETU+'_ii_sim_'+str(I_SIM_THRESHOLD)+'.pkl', 'rb') as f:
            item_sim_df = pickle.load(f)
        print(DATASETU, "_ii_sim_", str(I_SIM_THRESHOLD), " loaded")

    # Part-6 : If needed, get ranked neighborhood users / items
    if NEED_RANKED:
        if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_uu_sim_'+str(U_SIM_THRESHOLD)+'_ranked.pkl'):
            user_sim_dict = {}
            for i in tqdm.tqdm(range(1, num_users+1)):
                if i in user_clk_dict.keys():
                    i_clk = user_clk_dict[i]
                    i_neigh, i_neigh_score = [], []
                    for j in range(1, num_users+1):
                        if j in user_clk_dict.keys() and i != j:
                            j_clk = user_clk_dict[j]
                            num_common = len(set(i_clk) & set(j_clk))
                            num_base = len(set(i_clk) | set(j_clk))
                            score = num_common / num_base
                            if score >= U_SIM_THRESHOLD:
                                i_neigh.append(j)
                                i_neigh_score.append(score)

                    if len(i_neigh) >= U_NEIGH_NUM:
                        i_neigh_score, i_neigh = (list(x) for x in zip(*sorted(zip(i_neigh_score, i_neigh))))
                        i_neigh = i_neigh[-U_NEIGH_NUM:]
                    else:
                        i_neigh.extend([0]*(U_NEIGH_NUM-len(i_neigh)))

                    user_sim_dict[i] = i_neigh

            with open('data_'+DATASETL+'/'+DATASETU+'_uu_sim_'+str(U_SIM_THRESHOLD)+'_ranked.pkl', 'wb') as f:
                pickle.dump(user_sim_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('data_'+DATASETL+'/'+DATASETU+'_uu_sim_'+str(U_SIM_THRESHOLD)+'_ranked.pkl', 'rb') as f:
                user_sim_dict = pickle.load(f)
            print(DATASETU, "_uu_sim_", str(U_SIM_THRESHOLD), " ranked loaded")

        # item to item graph neighborhood
        if not os.path.exists('data_'+DATASETL+'/'+DATASETU+'_ii_sim_'+str(I_SIM_THRESHOLD)+'_ranked.pkl'):
            item_sim_dict = {}
            for i in tqdm.tqdm(range(1, num_items+1)):
                if i in item_clk_dict.keys():
                    i_clk = item_clk_dict[i]
                    i_neigh, i_neigh_score = [], []
                    for j in range(1, num_items+1):
                        if j in item_clk_dict.keys() and i != j:
                            j_clk = item_clk_dict[j]
                            num_common = len(set(i_clk) & set(j_clk))
                            num_base = len(set(i_clk) | set(j_clk))
                            score = num_common / num_base
                            if score >= I_SIM_THRESHOLD:
                                i_neigh.append(j)
                                i_neigh_score.append(score)

                    if len(i_neigh) >= I_NEIGH_NUM:
                        i_neigh_score, i_neigh = (list(x) for x in zip(*sorted(zip(i_neigh_score, i_neigh))))
                        i_neigh = i_neigh[-I_NEIGH_NUM:]
                    else:
                        i_neigh.extend([0]*(I_NEIGH_NUM-len(i_neigh)))

                    item_sim_dict[i] = i_neigh

            with open('data_'+DATASETL+'/'+DATASETU+'_ii_sim_'+str(I_SIM_THRESHOLD)+'_ranked.pkl', 'wb') as f:
                pickle.dump(item_sim_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('data_'+DATASETL+'/'+DATASETU+'_ii_sim_'+str(I_SIM_THRESHOLD)+'_ranked.pkl', 'rb') as f:
                item_sim_dict = pickle.load(f)
            print(DATASETU, "_ii_sim_", str(I_SIM_THRESHOLD), " ranked loaded")

    print("Data preprocessing done")
