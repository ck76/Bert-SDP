# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset, project_name="ant"):
        self.model_name = 'bert_sdp'
        self.train_path = dataset + '/data/'+project_name+'/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/'+project_name+'/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/'+project_name+'/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/ant/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 128                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '/Users/test/Documents/GitHub/Bert-SDP/JavaBERT'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        # print(x)
        # TODO 后续拼接CNN和LSTM
        return out

dataset = '/Users/test/Documents/GitHub/Bert-SDP/PROMISE'  # 数据集
net = Model(Config(dataset))
print(net)

x=torch.rand(10,256).long()
seq_len=torch.randn(10).long()
mask=torch.randn(10,256).long()

out = net((x,seq_len,mask))
print(out)