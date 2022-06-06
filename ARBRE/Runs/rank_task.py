proj_path = "/home/data/bch/oyy/ARBRE_pytorch/ARBRE/DatasetsLoad"  # change to your path

import sys
import os

sys.path.append(proj_path)

import numpy as np
import random
import time
import torch
from torch.utils.data import DataLoader
import sample
import dataset

proj_path = "/home/data/bch/oyy/ARBRE_pytorch/ARBRE/Models/Model"  # change to your path
sys.path.append(proj_path)

import arbre


def get_engine(name, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    print("=========== model: " + name + " ===========")
    if name == 'arbre':
        return arbre.ArbreEngine(Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings)
    else:
        raise ValueError('unknow model name: ' + name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Run(DataSettings, ModelSettings, TrainSettings, ResultSettings,
        mode='train',
        timestamp=None, checkpoint=None):
    # Setting
    setup_seed(817)

    model_name = ModelSettings['model_name']
    save_dir = ResultSettings['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = save_dir + model_name[0].upper() + model_name[1:] + '/'
    epoch = eval(TrainSettings['epoch'])
    batch_size = eval(TrainSettings['batch_size'])
    s_batch_size = eval(TrainSettings['s_batch_size'])

    # Data
    Sampler = sample.SampleGenerator(DataSettings)
    graphs = Sampler.generateGraphs()

    print('User count: %d. Item count: %d. ' % (Sampler.user_num, Sampler.item_num))
    print('Without Negatives, Train count: %d. Validation count: %d. Test count: %d' % (
    Sampler.train_size, Sampler.val_size, Sampler.test_size))

    # Model
    Engine = get_engine(model_name, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings)

    # Mode
    if timestamp is None:
        timestamp = time.time()
    localtime = str(time.asctime(time.localtime(int(timestamp))))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(save_dir + model_name + "_" + str(int(timestamp)) + ".txt", "a") as f:
        model_save_dir = save_dir + 'files/' + str(int(timestamp)) + '/'

        f.write('\n\n\n' + '========== ' + localtime + " " + str(int(timestamp)) + ' ==========' + '\n')
        f.write(str(DataSettings) + '\n' + str(ModelSettings) + '\n' + str(TrainSettings) + '\n')
        f.write(str(Engine.model))
        f.write('\n')

        if mode == "train":  # train mode
            if checkpoint:
                Engine.model = torch.load(model_save_dir + model_name + '.pt').to(TrainSettings['device'])
                print("=========== Training Continue ===========")
            else:
                print("=========== Training Start ===========")
            val_hr, val_ndcg = 0, 0
            best_result = ""
            endure_count = 0
            early_stop_step = 10
            pre_train_epoch = 0

            for epoch_i in range(0, epoch):

                # train
                Sampler.generateTrainNegative(combine=True)
                train_loader = DataLoader(dataset.PointDataset(Sampler.train_data), batch_size=batch_size, shuffle=True, num_workers=8)
                Engine.train(train_loader, graphs, epoch_i)

                # early stop
                if epoch_i >= pre_train_epoch:
                    # valid
                    val_df = Sampler.val_df.sample(frac=1.0)
                    val_neg = Sampler.val_neg.loc[Sampler.val_neg.user_id.isin(val_df.user_id.unique())]
                    test_pos_loader = DataLoader(dataset.RankDataset(val_df), batch_size=s_batch_size, shuffle=True, num_workers=8)
                    test_neg_loader = DataLoader(dataset.RankDataset(val_neg), batch_size=s_batch_size, shuffle=True, num_workers=8)
                    result, res = Engine.evaluate(test_pos_loader, test_neg_loader, graphs, epoch_i)
                    tmp_recall, tmp_ndcg = res

                    if tmp_ndcg > val_ndcg:
                        val_recall, val_ndcg = tmp_recall, tmp_ndcg
                        endure_count = 0
                        best_result = result
                        test_epoch = epoch_i
                        print(str(int(timestamp)) + ' new test result:', best_result)
                        # save log
                        f.write('epoch: ' + str(epoch_i) + '\n')
                        f.write(result + '\n')
                        # save model
                        if not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        torch.save(Engine.model, model_save_dir + model_name + '.pt')
                    else:
                        endure_count += 1

                    if endure_count > early_stop_step:
                        break

            # test
            print(str(int(timestamp)) + ' best test result:', best_result)
            f.write('best results(epoch: ' + str(test_epoch) + ' timestamp: ' + str(int(timestamp)) + '):\n' + best_result + '\n')

            print("Testing")
            Sampler._get_negative_sample()
            Engine.model = torch.load(model_save_dir + model_name + '.pt').to(TrainSettings['device'])
            Engine.eval_ks = [5, 10, 15]
            test_pos_loader = DataLoader(dataset.RankDataset(Sampler.test_df), batch_size=s_batch_size, shuffle=True, num_workers=8)
            test_neg_loader = DataLoader(dataset.RankDataset(Sampler.eval_neg), batch_size=s_batch_size, shuffle=True, num_workers=8)
            result, res = Engine.evaluate(test_pos_loader, test_neg_loader, graphs, epoch_id=0)

            print('test results( timestamp: ' + str(int(timestamp)) + '):\n', result)
            f.write('test results( timestamp: ' + str(int(timestamp)) + '):\n' + result + '\n')
        else:
            print("=========== Inference Start ===========")
            print("Testing")
            Sampler._get_negative_sample()

            Engine.model = torch.load(model_save_dir + model_name + '.pt').to(TrainSettings['device'])
            Engine.eval_ks = [5, 10, 15]
            test_pos_loader = DataLoader(dataset.RankDataset(Sampler.test_df), batch_size=s_batch_size, shuffle=True, num_workers=8)
            test_neg_loader = DataLoader(dataset.RankDataset(Sampler.eval_neg), batch_size=s_batch_size, shuffle=True, num_workers=8)
            result, res = Engine.evaluate(test_pos_loader, test_neg_loader, graphs, epoch_id=0)

            print('test results( timestamp: ' + str(int(timestamp)) + '):\n', result)
            f.write('test results( timestamp: ' + str(int(timestamp)) + '):\n' + result + '\n')
