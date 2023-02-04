# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset="PROMISE", project_name="ant"):
        self.model_name = 'bert_rnn_sdp'
        self.train_path = dataset + '/data/'+project_name+'/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/'+project_name+'/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/'+project_name+'/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/ant/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                        # 类别数
        self.num_epochs = 1                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = 'JavaBERT'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 512                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        #     Examples::
        #         >>> rnn = nn.LSTM(10, 20, 2)
        #         >>> input = torch.randn(5, 3, 10)
        #         >>> h0 = torch.randn(2, 3, 20)
        #         >>> c0 = torch.randn(2, 3, 20)
        #         >>> output, (hn, cn) = rnn(input, (h0, c0))
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        # Examples::
        #         >>> m = nn.Linear(20, 30)
        #         >>> input = torch.randn(128, 20)
        #         >>> output = m(input)
        #         >>> print(output.size())
        #         torch.Size([128, 30])
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    # todo x 是3个tensor
    def forward(self, x):
        # print(x)
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, (hidden, cell) = self.lstm(encoder_out)
        # attn_output = self.attention_net(output, hidden)
        # out = self.attention(out, hidden)
        out = self.dropout(out)
        out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out


dataset = '/Users/test/Documents/GitHub/Bert-SDP/PROMISE'  # 数据集
net = Model(Config(dataset))
print(net)

x=torch.rand(10,256).long()
seq_len=torch.randn(10).long()
mask=torch.randn(10,256).long()

out = net((x,seq_len,mask))
print(out)

#
# Time usage: 0:00:14
# Epoch [1/3]
# 0-------6
# (tensor([[  101, 23477, 10294,  ...,  8702,  2897,  2105],
#         [  101, 23477, 10294,  ..., 11192,  5521, 11813],
#         [  101, 23477, 10294,  ...,  3835, 15177,   168],
#         ...,
#         [  101, 23477, 10294,  ...,  3963,  1665, 17442],
#         [  101, 23477, 10294,  ...,  1891,   168,  1383],
#         [  101, 23477, 10294,  ...,     0,     0,     0]]), tensor([256, 256, 256, 256,  23, 256,  78, 256, 256,  82, 256,  23, 256, 256,
#         256, 256, 256,  31, 256, 186, 133, 256, 256, 228, 256, 256,  58,  47,
#         256, 256,  24, 256, 256, 256, 256,  30, 256, 256, 256, 256, 256, 256,
#         256, 210, 256, 256, 256,  98,  53,  84, 256, 256, 193, 140, 256, 256,
#         256, 256, 256, 182, 256, 256, 165, 256, 103,  22, 256, 256,  15,  77,
#         124, 256, 256,  55, 256,  96, 256, 256, 256, 256, 256, 256, 105, 132,
#         256, 256, 256, 256, 256, 256, 256, 256, 256,  96, 256, 244, 256, 256,
#         256, 256,  62, 126,  27, 256, 190, 256, 256, 256, 256, 256, 223, 136,
#         256, 256, 256, 256, 256, 256, 256, 256,  99, 256, 256,  23,  10, 256,
#         256,  30]), tensor([[1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         ...,
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 1, 1, 1],
#         [1, 1, 1,  ..., 0, 0, 0]]))