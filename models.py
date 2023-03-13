import torch
from torch import nn
import timm

class timm_pretrained_features(nn.Module):    
    def __init__(self, model):
        super(timm_pretrained_features, self).__init__()        
        self.net = timm.create_model(model, pretrained=True, num_classes=0)
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.net.eval()
    
    def get_features(self, x): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.net(x)
        return torch.flatten(fmap,
                             start_dim=1, end_dim=-1)