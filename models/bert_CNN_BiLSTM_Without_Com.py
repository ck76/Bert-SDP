# coding: UTF-8
import math

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
        self.model_name = 'bert_cnn_bilstm_without_com'
        self.train_path = dataset + '/data/' + project_name + '/train.txt'  # 训练集
        self.dev_path = dataset + '/data/' + project_name + '/dev.txt'  # 验证集
        self.test_path = dataset + '/data/' + project_name + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/ant/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2 # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 128  # mini-batch大小 todo 太大的话可能会导致我的电脑内存泄漏
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.0001  # 学习率
        self.bert_path = 'JavaBERT'
        # self.tokenizer =  AutoTokenizer.from_pretrained("CAUKiel/JavaBERT")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 512  # 卷积核数量(channels数)
        self.dropout = 0.2
        self.rnn_hidden = 300
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained(config.bert_path)
        # batch_first: If ``True``, then the input and output tensors are provided
        #             as `(batch, seq, feature)` 【8，】
        #             instead of `(seq, batch, feature)`.
        #             Note that this does not apply to hidden or cell states. See the
        #             Inputs/Outputs sections below for details.  Default: ``False``
        # todo  – input (batch, seq_len, feature)
        # todo  – output (seq_len, batch, num_directions * hidden_size)
        #
        #           input_size: The number of expected features in the input `x`
        #         hidden_size: The number of features in the hidden state `h`
        #         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        #             would mean stacking two LSTMs together to form a `stacked LSTM`,
        #             with the second LSTM taking in outputs of the first LSTM and
        #             computing the final results. Default: 1
        self.lstm = nn.LSTM(1, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.conv1 = nn.Conv2d(1, 1, (10, config.hidden_size))
        self.conv2 = nn.Conv2d(1, 1, (4, config.hidden_size))
        self.conv3 = nn.Conv2d(1, 1, (3, config.hidden_size))
        # self.conv2 = nn.Conv2d(1, config.num_filters, (5, config.hidden_size))
        for param in self.bert.parameters():
            param.requires_grad = False
        #     in_channels: int,
        #         out_channels: int,
        #         kernel_size: _size_2_t,
        #         stride: _size_2_t = 1,
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.rnn_hidden * config.num_layers, config.num_classes)

    # torch.Size([1, 1, 256, 768])
    def conv_and_pool(self, x, conv):
        # print(" conv_and_pool -1")
        # print(x.shape)
        x = conv(x)
        # print(" conv_and_pool -2")
        # print(x.shape)
        # x = F.relu(x)  # 就是压缩（维度减少，降维）
        # print(" conv_and_pool -3")
        # print(x.shape)
        # x = x.squeeze(3)
        # print(" conv_and_pool -4")
        # print(x.shape)
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # print(" conv_and_pool -5")
        # print(x.shape)
        return x

    # x: [batch, seq_len, hidden_dim*2]
    # query : [batch, seq_len, hidden_dim * 2]
    # 软注意力机制 (key=value=x)
    def attention_net(self, x, query, mask=None):

        d_k = query.size(-1)  # d_k为query的维度

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        #         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        #         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # print("1")  # torch.Size([1, 256, 768])
        # print(encoder_out.shape)
        out = encoder_out.unsqueeze(1)
        # print("2")  # torch.Size([1, 1, 256, 768])
        # print(out.shape)
        """
        in_channels
          这个很好理解，就是输入的四维张量[N, C, H, W]中的C了，即输入张量的channels数。这个形参是确定权重等可学习参数的shape所必需的。
        out_channels
          也很好理解，即期望的四维输出张量的channels数，不再多说。
        kernel_size
          卷积核的大小，一般我们会使用5x5、3x3这种左右两个数相同的卷积核，因此这种情况只需要写kernel_size = 5这样的就行了。如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）。
        stride = 1
          卷积核在图像窗口上每次平移的间隔，即所谓的步长。这个概念和Tensorflow等其他框架没什么区别，不再多言。
        原文链接：https://blog.csdn.net/qq_42079689/article/details/102642610
        """
        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out1 = self.conv_and_pool(out, self.conv1)
        # out2 = self.conv_and_pool(out, self.conv1)
        # out3 = self.conv_and_pool(out, self.conv1)
        # out = torch.cat((out1, out2, out3), 1)
        out = self.conv_and_pool(out, self.conv1)
        # print("3")  # torch.Size([1, 768])
        # print(out.shape)
        # # 序列长度seq_len=5, batch_size=3, 数据向量维数=10
        # input = torch.randn(5, 3, 10)
        # 论词的，一个词一行啊
        # 2、torch.randn(5, 3, 10) 数据中第一维度5（有5组数据，每组3行，每行10列）
        out = out.view(out.size(0), -1, out.size(-1))
        # print("4")  # torch.Size([1, 1, 768])
        # print(out.shape)
        out, (hidden, cell) = self.lstm(out)

        # ----------
        # out = self.dropout(out)

        query = self.dropout(out)
        # 加入attention机制
        attn_output, alpha_n = self.attention_net(out, query)
        # print("attn_output")
        # print(attn_output.shape)
        out = self.fc_cnn(attn_output)
        # ----------


        # print("5")  # torch.Size([1, 1, 1536])
        # print(out.shape)
        # out = self.fc_cnn(out)
        # print("6")
        # print(out.shape)
        return out


# net = Model(Config("../PROMISE"))
# print(net)
#
# # params = list(net.parameters())
# # print(len(params))
#
# x = torch.rand(1, 256).long()
# seq_len = torch.randn(1).long()
# mask = torch.randn(1, 256).long()
# print(type(x.shape))
# out = net((x, seq_len, mask))
# print(out)

#  (lstm): LSTM(768, 768, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
#   (convs): ModuleList(
#     (0): Conv2d(1, 256, kernel_size=(2, 768), stride=(1, 1))
#     (1): Conv2d(1, 256, kernel_size=(3, 768), stride=(1, 1))
#     (2): Conv2d(1, 256, kernel_size=(4, 768), stride=(1, 1))
#   )
#   (dropout): Dropout(p=0.1, inplace=False)
#   (fc_cnn): Linear(in_features=768, out_features=2, bias=True)
# ----------
#   (lstm): LSTM(768, 768, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
#   (convs): ModuleList(
#     (0): Conv2d(1, 256, kernel_size=(2, 768), stride=(1, 1))
#     (1): Conv2d(1, 256, kernel_size=(3, 768), stride=(1, 1))
#     (2): Conv2d(1, 256, kernel_size=(4, 768), stride=(1, 1))
#   )
#   (dropout): Dropout(p=0.1, inplace=False)
# todo
#   (fc_cnn): Linear(in_features=1536, out_features=2, bias=True)
# )
