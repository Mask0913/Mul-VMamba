import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from fvcore.nn import flop_count_table, FlopCountAnalysis

class MulMamba(BaseModel):
    def __init__(self, backbone: str = 'MulMamba-T', num_classes: int = 25, modals: list = ['img', 'depth', 'event', 'lidar']) -> None:
        super().__init__(backbone, num_classes, modals)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        y = self.backbone(x)
        y = self.decode_head(y) 
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

    
    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)

def load_dualpath_model(model, model_file):
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k.replace('patch_embed', 'rgb_path_embed')] = v
            state_dict[k.replace('patch_embed', 'extra_path_embed.0')] = v
            state_dict[k.replace('patch_embed', 'extra_path_embed.1')] = v
            state_dict[k.replace('patch_embed', 'extra_path_embed.2')] = v
        elif k.find('layers.0') >= 0:
            state_dict[k.replace('layers.0', 'rgb_block1')] = v
            state_dict[k.replace('layers.0', 'extra_block1.0')] = v
            state_dict[k.replace('layers.0', 'extra_block1.1')] = v
            state_dict[k.replace('layers.0', 'extra_block1.2')] = v
        elif k.find('layers.1') >= 0:
            state_dict[k.replace('layers.1', 'rgb_block2')] = v
            #state_dict[k.replace('layers.1', 'extra_block2')] = v
        elif k.find('layers.2') >= 0:
            state_dict[k.replace('layers.2', 'rgb_block3')] = v
            #state_dict[k.replace('layers.2', 'extra_block3')] = v
        elif k.find('layers.3') >= 0:
            state_dict[k.replace('layers.3', 'rgb_block4')] = v
            #state_dict[k.replace('layers.3', 'extra_block4')] = v

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    del state_dict

if __name__ == '__main__':
    # modals = ['img']
    modals = ['img', 'depth', 'event', 'lidar']
    model = MulMamba('MulMamba-T', 25, modals)
    model.init_pretrained('checkpoints/pretrained/Vmamba/VmambaTiny.pth')
    x = [torch.zeros(2, 3, 960, 960), torch.ones(2, 3, 960, 960), torch.ones(2, 3, 960, 960)*2, torch.ones(2, 3, 960, 960) *3]
    for i in range(len(x)):
        x[i] = x[i].cuda()
    model = model.cuda()
    for i in range(10):
        y = model(x)
        # 显存占用
        print(torch.cuda.memory_allocated() / 1024 ** 2)
        torch.clear_autocast_cache()
