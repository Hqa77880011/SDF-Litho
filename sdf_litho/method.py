import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,boundary,dice


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',32)
        self.encoder=Encoder(2,w);self.head=nn.Conv2d(w,1,1)
    def forward(self,b):
        x=b['x'].float();edge=boundary(x,1)
        field=self.head(self.encoder(torch.cat([x,edge],1)))
        probability=torch.sigmoid(-field/self.c.get('softness',1.))
        return {'sdf':field,'probability':probability}
    def objective(self,b,p):
        if self.c['variant']=='baseline':return F.binary_cross_entropy(p['probability'],b['y'].float())+dice(p['probability'],b['y'])
        target=b['sdf'].float();error=abs(p['sdf']-target)
        weight=1+self.c.get('edge_alpha',4)*torch.exp(-abs(target)/self.c.get('tau',3))
        if self.c['variant']=='no_weight':weight=torch.ones_like(weight)
        loss=(weight*error).mean()+dice(p['probability'],b['y'])
        edge=boundary(b['y'].float(),1)
        if self.c['variant']!='no_edge':loss=loss+.5*(edge*abs(p['sdf'])).sum()/edge.sum().clamp_min(1)
        if self.c['variant']!='no_corner':loss=loss+.2*(b['corner']*error).sum()/b['corner'].sum().clamp_min(1)
        return loss


def demo(seed,n,c):
    from scipy.ndimage import gaussian_filter,distance_transform_edt,binary_closing
    rng=np.random.default_rng(seed);x=np.zeros((n,1,32,32),np.float32)
    for image in x:
        for _ in range(5):
            row,col=rng.integers(1,24,2);image[:,row:row+5,col:col+8]=1
    y=np.array([[binary_closing(gaussian_filter(image[0],1.)>.45)] for image in x],np.float32)
    sdf=np.array([[distance_transform_edt(1-image[0])-distance_transform_edt(image[0])] for image in y],np.float32)
    corner=np.array([[abs(np.gradient(np.gradient(image[0],axis=0),axis=1))] for image in y],np.float32)
    return {'x':x,'y':y,'sdf':sdf,'corner':corner,'group':np.arange(n)}


def measure(p,d,c):
    edge=abs(d['sdf'])<=1.01
    return {'sdf_mae':float(np.mean(abs(p['sdf']-d['sdf']))),'edge_position_error_px':float(np.mean(abs(p['sdf'][edge])))}
