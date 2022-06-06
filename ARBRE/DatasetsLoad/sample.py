import torch
import time
import random
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy

import dgl


class SampleGenerator(object):
    """ Construct dataset """

    def __init__(self, DataSettings):
        # data settings
        self.DataSettings = DataSettings
        
        # data path
        start_time = time.time()
        self.data_dir = DataSettings['data_dir']
        self.data_name = DataSettings['data_name']
        self.data_path = self.data_dir + 'data_' + self.data_name[0].lower() + self.data_name[1:] + '/'
        
        # data settings
        self.train_neg_num = eval(DataSettings['train_neg_num'])
        self.eval_neg_num = eval(DataSettings['eval_neg_num'])
        self.graph_types = eval(DataSettings['graph_types'])

        # read data
        print("=========== reading", self.data_name, "data ===========")
        self._get_main_data()
        self._get_negative_sample()

        print(f'read pickle file cost {time.time()-start_time} seconds') 
    
   
    def _get_main_data(self):

        with open(self.data_path+self.data_name+'_info.pkl', 'rb') as f:
            user_set = pickle.load(f)
            item_set = pickle.load(f)
            sample_set = pickle.load(f)
            (num_users, num_items) = pickle.load(f)
            self.user_num, self.item_num = num_users, num_items

        with open(self.data_path+self.data_name+'_train_val_test.pkl', 'rb') as f:
            train_pos_set = pickle.load(f)
            val_pos_set = pickle.load(f)
            test_pos_set = pickle.load(f)

            self.train_df = pd.DataFrame(train_pos_set, columns=['user_id', 'item_id'])
            self.val_df = pd.DataFrame(val_pos_set, columns=['user_id', 'item_id'])
            self.test_df = pd.DataFrame(test_pos_set, columns=['user_id', 'item_id'])

        self.need_ranked = self.DataSettings['need_ranked']
        if 'user_sim_g' in self.graph_types:
            uu_sim_limit = self.DataSettings['uu_sim_limit']
            self.uu_sim_df = pd.read_pickle(self.data_path+self.data_name+'_uu_sim_'+uu_sim_limit+'.pkl')
            self.uu_sim_dict = self.uu_sim_df.groupby('user1')['user2'].apply(list).to_dict()
            if self.need_ranked:
                self.uu_sim_ranked = pd.read_pickle(self.data_path+self.data_name+'_uu_sim_'+uu_sim_limit+'_ranked.pkl')
        if 'item_sim_g' in self.graph_types:
            ii_sim_limit = self.DataSettings['ii_sim_limit']
            self.ii_sim_df = pd.read_pickle(self.data_path+self.data_name+'_ii_sim_'+ii_sim_limit+'.pkl')
            self.ii_sim_dict = self.ii_sim_df.groupby('item1')['item2'].apply(list).to_dict()
            if self.need_ranked:
                self.ii_sim_ranked = pd.read_pickle(self.data_path+self.data_name+'_ii_sim_'+ii_sim_limit+'_ranked.pkl')

        with open(self.data_path+'/'+self.data_name+'_ui_clk.pkl', 'rb') as f:
            user_clk_dict = pickle.load(f)
            item_clk_dict = pickle.load(f)
            self.user_clk_dict = user_clk_dict
            self.item_clk_dict = item_clk_dict

        self.train_size, self.val_size, self.test_size = self.train_df.shape[0], self.val_df.shape[0], self.test_df.shape[0]

    def _get_negative_sample(self):
        data_file_val = open(self.data_path+'/'+self.data_name+'_val_negative_'+str(self.eval_neg_num)+'.pkl', 'rb')
        neg_dict_val = pickle.load(data_file_val)
        data_file_eval = open(self.data_path+'/'+self.data_name+'_eval_negative_'+str(self.eval_neg_num)+'.pkl', 'rb')
        neg_dict_eval = pickle.load(data_file_eval)

        eval_neg_users, eval_neg_items = [], []
        users = self.test_df['user_id'].unique()
        for u in users:
            tmp_items = neg_dict_eval[u]
            eval_neg_users.extend([u]*len(tmp_items))
            eval_neg_items.extend(tmp_items)
        self.eval_neg = pd.DataFrame({'user_id': eval_neg_users, 'item_id': eval_neg_items})

        val_neg_users, val_neg_items = [], []
        users = self.val_df['user_id'].unique()
        for u in users:
            tmp_items = neg_dict_val[u]
            val_neg_users.extend([u]*len(tmp_items))
            val_neg_items.extend(tmp_items)
        self.val_neg = pd.DataFrame({'user_id': val_neg_users, 'item_id': val_neg_items})

    def generateTrainNegative(self, combine=True):
        bias_id = 1
        num_negatives = self.train_neg_num
        num_items = self.item_num
        neg_users, neg_items = [], []
        for row in self.train_df.iterrows():
            u, i = row[1]['user_id'], row[1]['item_id']
            for _ in range(num_negatives):
                j = np.random.randint(bias_id, num_items+bias_id)
                while j in self.user_clk_dict[u]:
                    j = np.random.randint(bias_id, num_items+bias_id)
                neg_users.append(u)
                neg_items.append(j)
        train_neg = pd.DataFrame({'user_id':neg_users, 'item_id':neg_items})
        train_neg['rating'] = 0
        train_pos = deepcopy(self.train_df)
        train_pos['rating'] = 1
        self.train_data = pd.concat([train_pos, train_neg], ignore_index=True)

    def _sample_graphs(self, u, v, sample_num):
        node_dict = pd.DataFrame({'u': u, 'v': v}).groupby('u')['v'].apply(list).to_dict()
        new_u, new_v = [], []
        for u in node_dict.keys():
            tmp_vs = node_dict[u]
            tmp_vs = random.sample(tmp_vs, min(len(tmp_vs), sample_num) )
            new_u.extend([u]*len(tmp_vs))
            new_v.extend(tmp_vs)
        return new_u, new_v

    def generateGraphs(self):
        graphs = {}
        for graph_type in self.graph_types:
            if graph_type == 'user_sim_g':
                u, v = self.uu_sim_df['user1'].tolist(), self.uu_sim_df['user2'].tolist()
                max_nodes = self.user_num+1
                user_sim_g = self._create_graph(u, v, max_nodes=max_nodes)
                user_sim_g.ndata['id'] = torch.LongTensor(np.arange(max_nodes))
                graphs['user_sim_g'] = user_sim_g
            elif graph_type == 'item_sim_g':
                u, v = self.ii_sim_df['item1'].tolist(), self.ii_sim_df['item2'].tolist()
                max_nodes = self.item_num+1
                item_sim_g = self._create_graph(u, v, max_nodes=max_nodes)
                item_sim_g.ndata['id'] = torch.LongTensor(np.arange(max_nodes))
                graphs['item_sim_g'] = item_sim_g
            else:
                raise ValueError('unknow graph type name: ' + graph_type)
        return graphs

    def _create_graph(self, u, v, max_nodes, self_loop=True):
        g = dgl.graph((u, v), num_nodes=max_nodes)
        if self_loop:
            g = dgl.transform.remove_self_loop(g)
            g.add_edges(g.nodes(), g.nodes())

        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        return g
    