#%%
import torch
import torch.nn as nn
from torchvision import models

'''
Implementation of Barlow Twins (https://arxiv.org/abs/2103.03230), adapted from (https://github.com/MaxLikesMath/Barlow-Twins-Pytorch) 
which it is adapted for ease of use for experiments from (https://github.com/facebookresearch/barlowtwins), 
with some modifications using code from (https://github.com/lucidrains/byol-pytorch)
'''

def flatten(t):
    return t.reshape(t.shape[0], -1)

class NetWrapper(nn.Module):
    # from https://github.com/lucidrains/byol-pytorch
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)

        return representation





class BarlowTwins(nn.Module):
    '''
    Adapted from https://github.com/facebookresearch/barlowtwins for arbitrary backbones, and arbitrary choice of which
    latent representation to use. Designed for models which can fit on a single GPU (though training can be parallelized
    across multiple as with any other model). Support for larger models can be done easily for individual use cases by
    by following PyTorch's model parallelism best practices.
    '''

    def __init__(self, lambd, scale_factor=1):
        '''
        :param lambd: tradeoff function
        :param scale_factor: Factor to scale loss by, default is 1
        '''
        super().__init__()

        self.lambd = lambd
        self.scale_factor = scale_factor
  

    def add_backbone(self, backbone, latent_id = -2, monochanel = False, backbone_name=None, replacements=None, verbose=False):
        '''
        :param backbone: Model backbone
        :param latent_id: name (or index) of the layer to be fed to the projection MLP
        :param monochanel: adapt backbone input to accept single channel images
        '''
        self.backbone = backbone
        self.latent_id = latent_id
        self.monochanel = monochanel
        self.backbone_name = backbone_name

        #Converting the input channels from 3 to 1
        if self.monochanel and "resnet" in self.backbone_name.lower():
            self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        elif self.monochanel and  any(ext in self.backbone_name.lower() for ext in ["b0", "b1", "b2"]):
            self.backbone.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
        elif self.monochanel and "b3" in self.backbone_name.lower():
            self.backbone.features[0][0] = nn.Conv2d(1, 40, 3, 2, 1, bias=False)
        elif self.monochanel and "b4" in self.backbone_name.lower():
            self.backbone.features[0][0] = nn.Conv2d(1, 48, 3, 2, 1, bias=False)
        
        #Removing the backbone layers until latent id
        if self.latent_id:
            self.backbone = NetWrapper(self.backbone, latent_id)
        #Print model structure
        if verbose:
            print("\n"+10*"="+str(self.backbone_name)+10*"=")
            print(self.backbone)
            print(30*"="+"\n")

    def add_projector(self,  projector_sizes, verbose = False):
        '''
        :param projector: size of the hidden layers in the projection MLP
        '''
        sizes = projector_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        #Print model structure
        if verbose:
            print("\n"+10*"="+"Projector"+10*"=")
            print(self.projector)
            print(30*"="+"\n")

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()        

    def barlow_loss(self, z1, z2):
        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])
        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor*(on_diag + self.lambd * off_diag)
        return(loss, on_diag, off_diag)

    def forward(self, y1, y2):
        #Forward through the backbone
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        #Forward through the projector
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        #Computing the loss from the latent vectors
        loss, on_diag, off_diag = self.barlow_loss(z1, z2)

        return loss, on_diag, off_diag


if __name__=='__main__':
    import torchvision

    model = torchvision.models.resnet18(zero_init_residual=True)
    model = models.efficientnet_b1(pretrained=False, zero_init_residual=True)
    twins = BarlowTwins(0.5)
    twins.add_backbone( 
                        backbone =model, 
                        latent_id = -2,#"layer4", 
                        monochanel = True, 
                        backbone_name='B1', 
                        verbose=True)
    twins.add_projector(
                        projector_sizes = [ ], 
                        verbose = True)

    inp1 = torch.rand(5,1,512,512)
    inp2 = torch.rand(5,1,512,512)
    twins.train()
    loss, on_diag, off_diag =twins(inp1, inp2)
    #model = model_utils.extract_latent.LatentHook(model, ['avgpool'])
    #out, dicti = model(inp1)
    print(loss, on_diag, off_diag)

# %%
