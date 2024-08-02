import torch.nn as nn
from models.morn import MORN
from models.asrn_res import ASRN

class MORAN(nn.Module):
    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False, 
                 inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)  # nclass = 2 for binary classification

    def forward(self, x, length=None, text=None, text_rev=None, test=False, debug=False):
        x_rectified = self.MORN(x, test, debug=debug)
        preds = self.ASRN(x_rectified, length, text, text_rev, test)
        return preds
