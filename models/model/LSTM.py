import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lstm=nn.LSTM(10,20,2,bidirectional=True)
        self.fc_lstm=nn.Linear(20,2)

    def forward(self, x):
        h0 = torch.randn(4, 3, 20)
        c0 = torch.randn(4, 3, 20)
        # out,  (hidden, cell)=self.lstm(x,(h0,c0))
        out, (hidden, cell) = self.lstm(x)
        print(out.shape)#torch.Size([5, 3, 40])
        print(hidden.shape)#torch.Size([4, 3, 20])
        print(cell.shape)#torch.Size([4, 3, 20])

        o=self.fc_lstm(out.view(-1,20))
        print(o.shape)
        return o

net = Net()
print(net)
# Net(
#   (lstm): LSTM(10, 20, num_layers=2, bidirectional=True)
# )
# len(params):16

# 一个模型可训练的参数可以通过调用 net.parameters() 返回：
params = list(net.parameters())#16
print("len(params):"+str(len(params)))

#         >>> rnn = nn.LSTM(10, 20, 2)
#         >>> input = torch.randn(5, 3, 10)
#         >>> h0 = torch.randn(2, 3, 20)
#         >>> c0 = torch.randn(2, 3, 20)
#         >>> output, (hn, cn) = rnn(input, (h0, c0))
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
out = net(input)
print(out)










# def __init__(self,in_channels,out_channels,hidden_size,num_layers,output_size,batch_size,seq_length) -> None:
#         super(convNet,self).__init__()
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.hidden_size=hidden_size
#         self.num_layers=num_layers
#         self.output_size=output_size
#         self.batch_size=batch_size
#         self.seq_length=seq_length
#         self.num_directions=1 # 单向LSTM
#         self.relu = nn.ReLU(inplace=True)
#         # (batch_size=64, seq_len=3, input_size=3) ---> permute(0, 2, 1)
#         # (64, 3, 3)
#         self.conv=nn.Sequential(
#             nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2), #shape(7,--)  ->(64,3,2)
#             nn.ReLU())
#         self.lstm=nn.LSTM(input_size=out_channels,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)