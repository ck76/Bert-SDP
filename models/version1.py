# todo 终于搞懂了

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 512  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 256
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # TODO 应该是在这更新了bert
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        # TODO 思考一下怎么拼一个CNN上去
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        # self.maxpool = nn.MaxPool1d(config.pad_size) 目前是1280  ---》  2
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, 2)

    # def forward(self, ids, mask, token_type_ids):
    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt")
        # print(inputs)
        encoder_out = self.bert(**inputs).pooler_output
        print(encoder_out.shape)
        print(encoder_out.size())
        out, _ = self.lstm(encoder_out)
        print("latm out------" + str(out.shape))
        print(out)
        out = torch.cat((encoder_out, out), 1)
        out = F.relu(out)
        # out = out.permute(0, 2, 1)
        # out = self.maxpool(out).squeeze()
        out = self.fc(out)
        print("final out------" + str(out.shape))
        print(out)
        return out


print("hee")
dataset = 'PROMISE'  # 数据集
config = Config(dataset)
model = Model(config)
print("hee2222")
# print(model)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("asad dasjknvgvghv  kjnkjnjk dasj")
# print(inputs)
model.forward("asad dasjkn dasj")
