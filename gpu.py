

import torch


print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)


#  pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu117
#
# https://pytorch.org/get-started/locally/#windows-pip

# nvidia-smi