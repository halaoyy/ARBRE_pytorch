proj_path = "./home/data/bch/oyy/ARBRE_pytorch/ARBRE/Models"
import sys
sys.path.append(proj_path)

import Models.engine as engine
import Models.utils.layer as layer
import Models.Graph.utils as utils

import numpy as np
import pandas as pd
import tqdm
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


class ArbreNet(nn.Module):
    def __init__(self, Sampler, ModelSettings):
        super().__init__()

        self.num_item, self.num_user = Sampler.item_num, Sampler.user_num
        self.user_clk = Sampler.user_clk_dict
        self.item_clk = Sampler.item_clk_dict
        self.hid_dim = eval(ModelSettings['hidden_dim'])
        embed_dim = eval(ModelSettings['embed_dim'])
        self.num_layer = eval(ModelSettings['num_layer'])
        attn_drop = eval(ModelSettings['attn_drop'])

        self.user_embedding = nn.Embedding(self.num_user+1, embed_dim)
        self.item_embedding = nn.Embedding(self.num_item+1, embed_dim)
        
        self.user_sim_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.user_sim_GNNs.append(utils.Aggregator(attn_drop))

        self.item_sim_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.item_sim_GNNs.append(utils.Aggregator(attn_drop))

        self.uu_sim_att = layer.Attention(ModelSettings)
        self.ii_sim_att = layer.Attention(ModelSettings)

        s_dim = 48
        self.Predictor_1 = layer.Predictor(self.hid_dim, s_dim)
        self.Predictor_2 = layer.Predictor(self.hid_dim, s_dim)
        self.Predictor_3 = layer.Predictor(self.hid_dim, s_dim)
        self.Predictor_4 = layer.Predictor(self.hid_dim, s_dim)

        self.FFN = layer.FFN(self.hid_dim)

        self.Multihead_SelfAttention = torch.nn.MultiheadAttention(self.hid_dim, num_heads=2, dropout=attn_drop)
        self.Multihead_ModuAttention = torch.nn.MultiheadAttention(self.hid_dim, num_heads=1, dropout=attn_drop)

        self.init_weights()

    def graph_aggregate(self, g, GNNs, node_embedding, mode='train', Type=''):
        g = g.local_var()
        init_embed = node_embedding
        all_embed = [init_embed]
        for l in range(self.num_layer):
            GNN_layer = GNNs[l]
            init_embed = GNN_layer(mode, g, init_embed)
            norm_embed = F.normalize(init_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.stack(all_embed)
        all_embed = torch.mean(all_embed, dim=0)
        return all_embed

    def get_item_attraction(self, target_user_embed, node_embeddings):
        item_users_pad = torch.LongTensor(self.item_users).to(target_user_embed.device)
        item_users_embed = nn.functional.embedding(item_users_pad, node_embeddings)  # [B, R, H]

        target_user_embed = target_user_embed.unsqueeze(1)
        target_user_embed = target_user_embed.repeat(1, item_users_embed.shape[1], 1)  # [B, R, H]
        item_modu_embeddings = torch.max(item_users_embed * target_user_embed, dim=1)[0]  # [B, H]

        return item_modu_embeddings

    def get_item_fusion(self, active_item_embed, target_users_embed, node_embeddings):
        batch_size = active_item_embed.shape[0]

        f_users = torch.LongTensor(self.ii_sim_users).to(active_item_embed.device)  # [B, F, R]
        f_users_embed = nn.functional.embedding(f_users, node_embeddings)  # [B, F, R, H]
        target_users_embed = target_users_embed.view(batch_size, 1, 1, -1)  # [B, 1, 1, H]
        target_users_embed = target_users_embed.repeat(1, f_users.shape[1], f_users.shape[2], 1)
        f_users_embed = torch.max(f_users_embed * target_users_embed, dim=2)[0]  # [B, F, H]

        ii_sim_lens = deepcopy(self.ii_sim_lens)
        a = self.ii_sim_att(active_item_embed, f_users_embed, torch.LongTensor(ii_sim_lens).to(target_users_embed.device))  # [B, F]
        a = torch.reshape(a, [batch_size, 1, -1])  # [B, 1, F]
        f_users_embed = torch.bmm(a, f_users_embed).squeeze()  # [B, 1, F] * [B, F, H] = [B, H]

        return f_users_embed

    def get_user_interest(self, target_item_embed, node_embeddings):
        user_items_pad = torch.LongTensor(self.user_items).to(target_item_embed.device)
        user_items_embed = nn.functional.embedding(user_items_pad, node_embeddings)  # [B, R, H]
        user_items_embed = user_items_embed.permute(1, 0, 2)  # [R, B, H]

        user_items_embed_, _ = self.Multihead_SelfAttention(user_items_embed, user_items_embed, user_items_embed)
        user_items_embed = self.FFN(user_items_embed, user_items_embed_)

        target_item_embed = target_item_embed.unsqueeze(0)
        user_modu_embeddings, _ = self.Multihead_ModuAttention(target_item_embed, user_items_embed, user_items_embed)
        user_modu_embeddings = user_modu_embeddings.squeeze(0)  # [B, H]

        return user_modu_embeddings
   
    def get_user_fusion(self, active_user_embed, target_items_embed, node_embeddings):
        batch_size = active_user_embed.shape[0]
        n_items = torch.LongTensor(self.uu_sim_items).to(active_user_embed.device)    # [B, F, R]

        # element-wise & max pooling method
        n_items_embed = nn.functional.embedding(n_items, node_embeddings)   # [B, F, R, D]
        target_items_embed = target_items_embed.view(batch_size, 1, 1, -1)  # [B, 1, 1, D]
        n_items_embed = torch.max(n_items_embed * target_items_embed.repeat(1, n_items.shape[1], n_items.shape[2], 1), dim=2)[0]    # [B, F, D]

        uu_sim_lens = deepcopy(self.uu_sim_lens)
        a = self.uu_sim_att(active_user_embed, n_items_embed, torch.LongTensor(uu_sim_lens).to(target_items_embed.device)) #[B, F]
        a = torch.reshape(a, [batch_size, 1, -1])  # [B, 1, F]
        n_items_embed = torch.bmm(a, n_items_embed).squeeze()  # [B, 1, F] * [B, F, H] = [B, H]

        return n_items_embed

    def forward(self, user, item, user_sim_g, item_sim_g, mode):
        user = user.squeeze()
        item = item.squeeze()
        
        # ------------------ Embedding layer -----------------
        user_id_embedding = self.user_embedding(user_sim_g.ndata['id'])
        item_id_embedding = self.item_embedding(item_sim_g.ndata['id'])

        # ----------------Graph Aggregation Layer-------------
        # user collaborative graph aggregation
        user_preference_embedding = self.graph_aggregate(user_sim_g, self.user_sim_GNNs, user_id_embedding, Type='user_sim')
        user_preference_embedding[0] = torch.zeros_like(user_preference_embedding[0])
        # item collaborative graph aggregation
        item_attribute_embedding = self.graph_aggregate(item_sim_g, self.item_sim_GNNs, item_id_embedding, Type='item_sim')
        item_attribute_embedding[0] = torch.zeros_like(item_attribute_embedding[0])

        user_preference = user_preference_embedding[user]
        item_attribute = item_attribute_embedding[item]

        # ---------Asymmetrical Context-aware Modulation-------
        # item domain
        item_attraction = self.get_item_attraction(user_preference, user_preference_embedding)
        item_neigh_attraction = self.get_item_fusion(item_attraction, user_preference, user_preference_embedding)
        item_attraction = 0.5 * (item_attraction + item_neigh_attraction)

        # user domain
        user_interest = self.get_user_interest(item_attribute, item_attribute_embedding)
        user_neigh_interest = self.get_user_fusion(user_interest, item_attribute, item_attribute_embedding)
        user_interest = 0.5*(user_interest + user_neigh_interest)

        # ------------------Prediction Layer-------------------
        scores_list = [
            self.Predictor_1(user_preference, item_attribute),
            self.Predictor_2(user_interest, item_attribute),
            self.Predictor_3(user_preference, item_attraction),
            self.Predictor_4(user_interest, item_attraction),
        ]

        if mode == 'train':
            return scores_list
        else:
            scores_list = [x.view(-1, 1) for x in scores_list]
            return torch.mean(torch.cat(scores_list, dim=1), dim=1)

    def init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)


class ArbreEngine(engine.Engine):
    
    def __init__(self, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings):
        self.Sampler = Sampler
        self.model = ArbreNet(Sampler, ModelSettings)
        self.model.to(TrainSettings['device'])
        self.batch_size = eval(TrainSettings['batch_size'])
        self.eval_neg_num = eval(DataSettings['eval_neg_num'])
        self.eval_ks = eval(TrainSettings['eval_ks'])
        self.user_clk_sample_num = eval(DataSettings['user_clk_sample_num'])
        self.item_clk_sample_num = eval(DataSettings['item_clk_sample_num'])
        self.user_neigh_sample_num = eval(DataSettings['user_neigh_sample_num'])
        self.item_neigh_sample_num = eval(DataSettings['item_neigh_sample_num'])

        super(ArbreEngine, self).__init__(TrainSettings, ModelSettings, ResultSettings)

        self.user_clk = Sampler.user_clk_dict
        self.item_clk = Sampler.item_clk_dict
        self.ii_sim = Sampler.ii_sim_dict
        self.uu_sim = Sampler.uu_sim_dict
        if DataSettings['need_ranked']:
            self.ii_sim_ranked = Sampler.ii_sim_ranked
            self.uu_sim_ranked = Sampler.uu_sim_ranked

    def history_sample(self, ks, clk_dict, masks, mode='train', Type='user'):
        k_clks = [[] if x not in clk_dict.keys() else deepcopy(clk_dict[x]) for x in ks]
        if Type == 'user':
            h_sample_num = self.user_clk_sample_num
        else:
            h_sample_num = self.item_clk_sample_num

        clk_lens = []
        r_max = 0
        for i in range(len(k_clks)):
            mask = masks[i]
            if mask in k_clks[i]:
                k_clks[i].remove(mask)

            if not k_clks[i]:
                k_clks[i] = [0]
            r_max = max(r_max, len(k_clks[i]))
            clk_lens.append(len(k_clks[i]))

        # padding
        for i in range(len(k_clks)):
            cur_len = len(k_clks[i])
            k_clks[i].extend([0]*(r_max-cur_len))

        return k_clks, clk_lens

    def ii_sim_sample(self, items, item_sim, item_clk, is_sample=True):
        n_sample_num = self.item_neigh_sample_num
        h_sample_num = self.item_clk_sample_num
        batch_size = len(items)
        i_neighs = [[] if x not in item_sim.keys() else deepcopy(item_sim[x]) for x in items]
        neighs_users = []
        n_max, r_max = 0, 0
        neighs_lens = []
        for i in range(batch_size):
            cur_n_users = []

            cur_neigh = i_neighs[i]
            if is_sample:
                sample_num = min(n_sample_num, len(cur_neigh))
                cur_neigh = random.sample(cur_neigh, sample_num)
            n_max = max(len(cur_neigh), n_max)

            for f in cur_neigh:
                if f in item_clk.keys():
                    tmp_users = item_clk[f]
                    if is_sample:
                        sample_num = min(h_sample_num, len(tmp_users))
                        tmp_users = random.sample(tmp_users, sample_num)
                    r_max = max(len(tmp_users), r_max)
                    cur_n_users.append(tmp_users)

            if not cur_n_users:
                cur_n_users = [[0]]
            neighs_users.append(cur_n_users)
            neighs_lens.append(len(cur_neigh))

        # padding
        for i in range(len(neighs_users)):
            cur_n_len = len(neighs_users[i])
            for j in range(cur_n_len):
                neighs_users[i][j].extend([0]*(r_max-len(neighs_users[i][j])))
            neighs_users[i].extend([[0]*r_max]*(n_max-j-1))

        return neighs_users, neighs_lens

    def uu_sim_sample(self, users, user_sim, user_clk, is_sample=True):
        n_sample_num = self.user_neigh_sample_num
        h_sample_num = self.user_clk_sample_num
        batch_size = len(users)
        u_neighs = [[] if x not in user_sim.keys() else deepcopy(user_sim[x]) for x in users]
        neighs_items = []
        n_max, r_max = 0, 0
        neighs_lens = []
        for i in range(batch_size):
            cur_n_items = []
            cur_neigh = u_neighs[i]
            if is_sample:
                sample_num = min(n_sample_num, len(cur_neigh))
                cur_neigh = random.sample(cur_neigh, sample_num)
            n_max = max(len(cur_neigh), n_max)

            for f in cur_neigh:
                if f in user_clk.keys():
                    tmp_items = user_clk[f]
                    if is_sample:
                        sample_num = min(h_sample_num, len(tmp_items))
                        tmp_items = random.sample(tmp_items, sample_num)
                    r_max = max(len(tmp_items), r_max)
                    cur_n_items.append(tmp_items)
            
            if not cur_n_items:
                cur_n_items = [[0]]
            neighs_items.append(cur_n_items)
            neighs_lens.append(len(cur_neigh))
        
        # padding
        for i in range(len(neighs_items)):
            cur_f_len = len(neighs_items[i])
            for j in range(cur_f_len):
                neighs_items[i][j].extend([0]*(r_max-len(neighs_items[i][j])))
            neighs_items[i].extend([[0]*r_max]*(n_max-j-1))
        
        return neighs_items, neighs_lens

    def train(self, train_loader, graphs, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        device = self.device
        user_sim_g = graphs['user_sim_g'].to(torch.device(device))
        item_sim_g = graphs['item_sim_g'].to(torch.device(device))

        total_loss = 0
        tmp_train_loss = []
        t0 = time.time()
        for i, input_list in enumerate(tqdm.tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)):
            batch_users, batch_items = list(input_list[1].numpy()), list(input_list[2].numpy())
            self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_clk, batch_users, mode='train', Type='item')
            self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_clk, batch_items, mode='train', Type='user')
            self.model.uu_sim_items, self.model.uu_sim_lens = self.uu_sim_sample(batch_users, self.uu_sim, self.user_clk)  # if needed, self.uu_sim can be replaced by self.uu_sim_ranked
            self.model.ii_sim_users, self.model.ii_sim_lens = self.ii_sim_sample(batch_items, self.ii_sim, self.item_clk)

            # run model
            input_list = [x.to(device) for x in input_list]
            self.optimizer.zero_grad()
            pred_list = self.model(*input_list[1:], user_sim_g, item_sim_g, mode='train')
            # loss
            with torch.autograd.set_detect_anomaly(True):
                label = input_list[0]
                loss = 0
                for pred in pred_list:
                    loss += self.criterion(pred.squeeze(), label.float())
                loss.backward(retain_graph=False)
                self.optimizer.step()
            tmp_train_loss.append(loss.item())
            total_loss += loss.item()

        t1 = time.time()
        print("Epoch ", epoch_id, " Train cost:", t1-t0, " Loss: ", np.mean(tmp_train_loss))
        return np.mean(tmp_train_loss) 

    def evaluate(self, eval_pos_loader, eval_neg_loader, graphs, epoch_id, mode='evaluate'):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        device = self.device
        user_sim_g = graphs['user_sim_g'].to(torch.device(device))
        item_sim_g = graphs['item_sim_g'].to(torch.device(device))

        # evaluate pos sample
        t0 = time.time()
        pos_users, pos_scores = [], []
        for i, input_list in enumerate(tqdm.tqdm(eval_pos_loader, desc="eval pos_s", smoothing=0, mininterval=1.0)):
            batch_users, batch_items = list(input_list[0].numpy()), list(input_list[1].numpy())
            self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_clk, batch_users, mode='eval', Type='item')
            self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_clk, batch_items, mode='eval', Type='user')
            self.model.uu_sim_items, self.model.uu_sim_lens = self.uu_sim_sample(batch_users, self.uu_sim, self.user_clk)
            self.model.ii_sim_users, self.model.ii_sim_lens = self.ii_sim_sample(batch_items, self.ii_sim, self.item_clk)

            input_list = [x.to(device) for x in input_list]
            pred = self.model(*input_list, user_sim_g, item_sim_g, mode='eval')
            pos_users.extend(batch_users)
            pos_scores.extend(list(pred.data.cpu().numpy()))

        t1 = time.time()
        print("Epoch ", epoch_id, " Test cost:", t1-t0)
        
        # evaluate neg sample
        t2 = time.time()
        neg_users, neg_scores = [], []
        for i, input_list in enumerate(tqdm.tqdm(eval_neg_loader, desc="eval neg_s", smoothing=0, mininterval=1.0)):
            batch_users, batch_items = list(input_list[0].numpy()), list(input_list[1].numpy())
            self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_clk, batch_users, mode='eval', Type='item')
            self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_clk, batch_items, mode='eval', Type='user')
            self.model.uu_sim_items, self.model.uu_sim_lens = self.uu_sim_sample(batch_users, self.uu_sim, self.user_clk)
            self.model.ii_sim_users, self.model.ii_sim_lens = self.ii_sim_sample(batch_items, self.ii_sim, self.item_clk)

            input_list = [x.to(device) for x in input_list]
            pred = self.model(*input_list, user_sim_g, item_sim_g, mode='eval')
            neg_users.extend(batch_users)
            neg_scores.extend(list(pred.data.cpu().numpy()))

        t3 = time.time()
        print("Epoch ", epoch_id, " Test cost:", t3-t2)

        pos_df = pd.DataFrame({'uid': pos_users, 'score': pos_scores})
        pos_df.sort_values(by=['uid'], ascending=False, inplace=True)
        pos_res = pos_df.groupby('uid')['score'].apply(list).to_dict()
        neg_df = pd.DataFrame({'uid': neg_users, 'score': neg_scores})
        neg_res = neg_df.groupby('uid')['score'].apply(list).to_dict()

        res_recall, res_ndcg, evaluate_result = self.get_metric(pos_res, neg_res, ks=self.eval_ks)

        select_k = self.eval_ks[2]
        print(mode, "pos cost: ", t1-t0, " neg cost: ", t3-t2, "; result, ", "Recall@"+str(select_k)+": ", res_recall[2], " NDCG@"+str(select_k)+": ", res_ndcg[2])
        return evaluate_result, [res_recall[2], res_ndcg[2]]

    