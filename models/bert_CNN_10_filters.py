# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset="PROMISE", project_name="ant"):
        self.model_name = 'bert_cnn'
        self.train_path = dataset + '/data/'+project_name+'/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/'+project_name+'/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/'+project_name+'/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        #
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                        # 类别数
        self.num_epochs = 1                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
#         # todo pad_size
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.0001                                       # 学习率
        self.bert_path = 'JavaBERT'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (5, 5, 5, 5, 5,5, 5, 5)                                   # 卷积核尺寸
        self.num_filters = 10                                          # 卷积核数量(channels数)
        self.dropout = 0.01
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        #     Examples:
        #         >>> # With square kernels and equal stride
        #         >>> m = nn.Conv2d(16, 33, 3, stride=2)
        #         >>> # non-square kernels and unequal stride and with padding
        #         >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        #         >>> # non-square kernels and unequal stride and with padding and dilation
        #         >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        #         >>> input = torch.randn(20, 16, 50, 100)
        #         >>> output = m(input)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # number=X.size  # 计算 X 中所有元素的个数
        # X_row=np.size(X,0)  #计算 X 的行数
        # X_col=np.size(X,1)  #计算 X 的列数
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # # Max pooling over a (2, 2) window
        #         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #         # If the size is a square you can only specify a single number
        #         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x

    def forward(self, x):
        context = x[0]  # todo 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        # C = torch.cat( (A,B),0 )  #按维数0拼接（竖着拼）
        # C = torch.cat( (A,B),1 )  #按维数1拼接（横着拼）
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        out = self.sigmoid(out)
        return out



# dataset = '/Users/test/Documents/GitHub/Bert-SDP/PROMISE'  # 数据集
# net = Model(Config(dataset))
# # print(net)
#
# # (convs): ModuleList(
# #     (0): Conv2d(1, 256, kernel_size=(2, 768), stride=(1, 1))
# #     (1): Conv2d(1, 256, kernel_size=(3, 768), stride=(1, 1))
# #     (2): Conv2d(1, 256, kernel_size=(4, 768), stride=(1, 1))
# #   )
# #   (dropout): Dropout(p=0.1, inplace=False)
# #   (fc_cnn): Linear(in_features=768, out_features=2, bias=True)
#
# x=torch.rand(10,256).long()
# seq_len=torch.randn(10).long()
# mask=torch.randn(10,256).long()
#
# out = net((x,seq_len,mask))
# print(out)