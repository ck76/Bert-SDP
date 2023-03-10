# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
import pandas as pd
import numpy as np


# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("CAUKiel/JavaBERT")
#
# model = AutoModelForMaskedLM.from_pretrained("CAUKiel/JavaBERT")
def pre_process_data(path):
    data = pd.read_csv(path)
    data = data[
        ['mfa', 'ic', 'cbm', 'rfc', 'dam', 'ce', 'cbo', 'moa', 'wmc', 'ca', 'dit', 'noc', 'lcom3', 'lcom', 'cam', 'amc',
         'npm', 'loc', 'bugs']]
    labels = np.array(data.iloc[:, [-1]])
    features = np.array(data.iloc[:, :-1])
    labels = np.where(labels > 0, 1, labels)
    return features.reshape((-1, 18, 1)), labels


class Config(object):
    """配置参数"""

    def __init__(self, dataset="PROMISE", project_name="ant"):
        self.model_name = 'bert_cnn_bilstm_sdp'
        self.train_path = dataset + '/data/ant/train.txt'  # 训练集
        self.dev_path = dataset + '/data/ant/dev.txt'  # 验证集
        self.test_path = dataset + '/data/ant/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/ant/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2 # 类别数
        self.num_epochs = 1  # epoch数
        self.batch_size = 128  # mini-batch大小 todo 太大的话可能会导致我的电脑内存泄漏
        self.pad_size = 256  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './JavaBERT'
        # self.tokenizer =  AutoTokenizer.from_pretrained("CAUKiel/JavaBERT")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 512  # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        print("11111")
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        print("2222222")
        out = encoder_out.unsqueeze(1)
        print(out.shape)  # torch.Size([64, 1, 256, 768])
        print("33333")
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        print(out.shape)  # torch.Size([64, 768])
        # print("444444")
        # out = self.dropout(out)
        # out = self.fc_cnn(out)
        # print("555555")
        # return out
        out = out.unsqueeze(0)
        print("444444")

        out, (hidden, cell) = self.lstm(out)
        print(out.shape)  # torch.Size([1, 64, 1536])
        print(hidden.shape)  # torch.Size([4, 1, 768])
        print(cell.shape)  # torch.Size([4, 1, 768])
        print("555555")
        out = out.view(-1, self.config.num_filters * len(self.config.filter_sizes))
        print("66666666")
        print(out.shape)  #
        o = self.fc_cnn(out)
        print("7777777")
        print(o.shape)
        return o


net = Model(Config("PROMISE"))
# print(net)
# (convs): ModuleList(
#     (0): Conv2d(1, 256, kernel_size=(2, 768), stride=(1, 1))
#     (1): Conv2d(1, 256, kernel_size=(3, 768), stride=(1, 1))
#     (2): Conv2d(1, 256, kernel_size=(4, 768), stride=(1, 1))
#   )
#   (dropout): Dropout(p=0.1, inplace=False)
#   (fc_cnn): Linear(in_features=768, out_features=2, bias=True)

params = list(net.parameters())
print(len(params))

# (x, seq_len, mask), y
# input = torch.randn(1, 1, 32, 32)
x = torch.randn(0, 23477)
x = torch.randint(0, 23477, (64, 256))
print(x.shape)
print(x)
seq_len = torch.randint(0, 256, (64,))
print(seq_len.shape)
print(seq_len)
mask = torch.randint(0, 1, (64, 256))
print(mask.shape)
print(mask)
y = torch.randint(0, 1, (64,))
input = (x, seq_len, mask)
out = net(input)
print(out)
