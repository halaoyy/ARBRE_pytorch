proj_path = "/home/data/bch/oyy/ARBRE_pytorch/ARBRE/"  # change to your path
import sys
import argparse

sys.path.append(proj_path)

import rank_task
from configparser import ConfigParser

cfg = ConfigParser()

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--timestamp', default=None, nargs='?', help='timestamp')
    parser.add_argument('--checkpoint', default=None, nargs='?', help='checkpoint')

    args = parser.parse_args()
    data = args.data_name
    model = args.model_name
    timestamp = args.timestamp
    checkpoint = args.checkpoint

    if timestamp is None or checkpoint == 'True':
        mode = 'train'
    else:
        mode = 'test'

    # get running setting
    cfg.read(proj_path + 'Runs/configurations/' + data + '/' + data + '_' + model + '.ini')

    print(model)

    # run
    rank_task.Run(DataSettings=dict(cfg.items("DataSettings")),
                  ModelSettings=dict(cfg.items("ModelSettings")),
                  TrainSettings=dict(cfg.items("TrainSettings")),
                  ResultSettings=dict(cfg.items("ResultSettings")),
                  mode=mode,
                  timestamp=timestamp,
                  checkpoint=checkpoint,
                  )
