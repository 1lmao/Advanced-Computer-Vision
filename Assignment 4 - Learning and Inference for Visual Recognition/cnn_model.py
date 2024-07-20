########################################################################
# 2. DEFINE YOUR CONVOLUTIONAL NEURAL NETWORK
########################################################################

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, init_weights=False):
        super(ConvNet, self).__init__()
        #INITIALIZE LAYERS HERE
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
		#PASS IMAGE X THORUGH EACH LAYER DEFINED ABOVE
        out = 
        return out

    def _initialize_weights(self):
        #INITIALIZE WEIGHTS

