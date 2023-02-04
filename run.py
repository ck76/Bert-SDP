# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()



if __name__ == '__main__':
    # dataset = '../PROMISE'  # 数据集
    project_name ="ant"
    # todo 从这里控制模型的选择
    # model_name = args.model  #
    model_name = "bert_CNN_BiLSTM_Without_Com"
    # model_name = "bert_CNN"
    # model_name = "bert_CNN_SDP"
    # model_name = "bert_RNN"
    # model_name = "bert_BiLSTM"
    # model_name = "bert_SDP"
    print(model_name)
    x = import_module('models.' + model_name)
    # config = x.Config(dataset)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # TODO 这里开始训练
    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
