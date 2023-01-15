import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lstm = nn.LSTM(10, 20, 2, bidirectional=True)
        self.fc_lstm = nn.Linear(20, 2)

    def forward(self, x):
        print(x.shape)  # torch.Size([1, 1, 32, 32])
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print(x.shape)  # torch.Size([1, 6, 14, 14])
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(x.shape)  # torch.Size([1, 16, 5, 5])
        # x = x.view(-1, self.num_flat_features(x))
        x = x.squeeze(0)  # torch.Size([16, 5, 5])
        print(x.shape)
        x = x.view(1, -1, 10)
        out, (hidden, cell) = self.lstm(x)
        print(out.shape)
        out = out.view(-1, 20)
        x = self.fc_lstm(out)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (lstm): LSTM(10, 20, num_layers=2, bidirectional=True)
#   (fc_lstm): Linear(in_features=20, out_features=2, bias=True)
# )