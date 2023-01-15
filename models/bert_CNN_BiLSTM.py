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
    data = data[['mfa', 'ic', 'cbm', 'rfc', 'dam', 'ce', 'cbo', 'moa', 'wmc', 'ca', 'dit', 'noc', 'lcom3', 'lcom', 'cam', 'amc', 'npm', 'loc', 'bugs']]
    labels = np.array(data.iloc[:, [-1]])
    features = np.array(data.iloc[:, :-1])
    labels = np.where(labels>0, 1, labels)
    return features.reshape((-1,18,1)), labels

class Config(object):

    """配置参数"""
    def __init__(self, dataset, project_name="ant"):
        self.model_name = 'bert_cnn_bilstm_sdp'
        self.train_path = dataset + '/data/'+project_name+'/train.txt'  # 训练集
        self.dev_path = dataset + '/data/'+project_name+'/dev.txt'  # 验证集
        self.test_path = dataset + '/data/'+project_name+'/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/ant/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 64                                           # mini-batch大小 todo 太大的话可能会导致我的电脑内存泄漏
        self.pad_size = 256                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '/Users/test/Documents/GitHub/Bert-SDP/JavaBERT'
        # self.tokenizer =  AutoTokenizer.from_pretrained("CAUKiel/JavaBERT")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


net = Model(Config("/Users/test/Documents/GitHub/Bert-SDP/PROMISE"))
# print(net)

# params = list(net.parameters())
# print(len(params))

x=torch.rand(10,256).long()
seq_len=torch.randn(10).long()
mask=torch.randn(10,256).long()

out = net((x,seq_len,mask))
print(out)