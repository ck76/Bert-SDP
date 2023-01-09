
import models.bert_CNN_BiLSTM
import models.bert_CNN
import models.bert_CNN_SDP
import models.bert
import models.bert_CNN_BiLSTM_Metrics
import models.bert_CNN_BiLSTM_Attention
import models.bert_CNN_BiLSTM_Metrics_Attention
import models.bert_RCNN_SDP
import models.bert_RNN
import models.bert_RNN_SDP
import models.bert_SDP
dataset = 'PROMISE'  # 数据集

# (convs): ModuleList(
#     (0): Conv2d(1, 256, kernel_size=(2, 768), stride=(1, 1))
#     (1): Conv2d(1, 256, kernel_size=(3, 768), stride=(1, 1))
#     (2): Conv2d(1, 256, kernel_size=(4, 768), stride=(1, 1))
#   )
#   (dropout): Dropout(p=0.1, inplace=False)
#   (fc_cnn): Linear(in_features=768, out_features=10, bias=True)
bert_cnn = models.bert_CNN.Model(models.bert_CNN.Config(dataset))
print(bert_cnn)

bert_cnn_bilstm = models.bert_CNN_BiLSTM.Model(models.bert_CNN_BiLSTM.Config(dataset))
print(bert_cnn_bilstm)

# (lstm): LSTM(768, 768, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
#   (dropout): Dropout(p=0.1, inplace=False)
#   (fc_rnn): Linear(in_features=1536, out_features=10, bias=True)
bert_rnn = models.bert_RNN.Model(models.bert_RNN.Config(dataset))
print(bert_rnn)

# (lstm): LSTM(768, 768, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
#   (dropout): Dropout(p=0.1, inplace=False)
#   (fc_rnn): Linear(in_features=1536, out_features=2, bias=True)
bert_rnn_sdp = models.bert_RNN_SDP.Model(models.bert_RNN_SDP.Config(dataset))
print(bert_rnn_sdp)