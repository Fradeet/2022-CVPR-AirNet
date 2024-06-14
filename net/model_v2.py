from torch import nn

from net.encoder import CBDE
from net.DGRN import DGRN_U as DGRN


class AirNet_v2(nn.Module):
    def __init__(self, opt):
        super(AirNet_v2, self).__init__()

        # Restorer
        self.R = DGRN(opt)

        # Encoder
        self.E = CBDE(opt)
        

    def forward(self, x_query, x_key):
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            restored = self.R(x_query, inter)

            return restored, logits, labels
        else:
            fea, inter = self.E(x_query, x_query)

            restored = self.R(x_query, inter)

            return restored
