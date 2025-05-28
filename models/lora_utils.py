from torchtune.modules.peft import LoRALinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from HoViT.models.HoViT import CNNDownsample, LevitBlock, NormLinear, Stem16

def convert_to_lora_model(model, rank=16, alpha=32, exclude=[]):
    for name, module in model.named_children():
        if name in exclude:
            continue
        if isinstance(module, nn.Linear):
            lora_linear = LoRALinear(
                in_dim=module.in_features,
                out_dim=module.out_features,
                rank=rank,
                alpha=alpha,
                use_bias=module.bias is not None
            )
            lora_linear.weight.data = module.weight.data #모델이 처음부터 다시 학습할 필요 없게 하기 위해
            if module.bias is not None:
                lora_linear.bias.data = module.bias.data
            lora_linear.to(module.weight.device)
            # 기존 linear layer를 loRA linear로 교체체
            setattr(model, name, lora_linear)
        # 재귀함수 (하위모듈도 확인)
        else:
            convert_to_lora_model(module, rank, alpha, exclude)

    return model

class LevitStage_TinyFusion(nn.Module):
    def __init__(self, dim, out_dim, num_heads, num_blocks, num_select, downsample=True):
        super(LevitStage_TinyFusion, self).__init__()
        assert num_select <= num_blocks
        self.downsample = CNNDownsample(dim, out_dim) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[LevitBlock(out_dim, num_heads) for _ in range(num_blocks)])
        self.num_blocks = num_blocks
        self.num_select = num_select
        init_probs = torch.ones(num_blocks) / num_blocks
        self.gumbel_gate = nn.Parameter(torch.log(init_probs))


    def forward(self, x, tau=1):
        x = self.downsample(x)

        if self.training:
            gate_probs = F.gumbel_softmax(self.gumbel_gate, tau=tau, hard=False)
        else:
            gate_probs = F.gumbel_softmax(self.gumbel_gate, tau=tau, hard=True)

        for i in range(self.num_blocks):
            if gate_probs[i] > 0: # skip zero blocks
              x = x + gate_probs[i] * self.blocks[i](x)
        return x
    
class LevitDistilledTinyfusion(nn.Module):
    def __init__(self, num_classes=9):
        super(LevitDistilledTinyfusion, self).__init__()

        self.stem = Stem16()

        self.stage1 = LevitStage_TinyFusion(dim=256, out_dim=256, num_heads=4, num_blocks=4, num_select=2, downsample=False) # block 수 적용
        self.stage2 = LevitStage_TinyFusion(dim=256, out_dim=384, num_heads=6, num_blocks=4, num_select=2, downsample=True)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.head = NormLinear(in_features=512, out_features=num_classes, dropout_prob=0.0)
        self.head_dist = NormLinear(in_features=512, out_features=num_classes, dropout_prob=0.0)

    def forward(self, x, tau):
        x = self.stem(x)

        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)
        x = self.stage1(x, tau)
        x = self.stage2(x, tau)

        H = W = int(x.shape[1]**0.5)
        x = x.transpose(1, 2).view(B, 384, H, W)

        x = self.conv1x1(x)

        x = torch.mean(x, dim=(2, 3))
        out = self.head(x)
        out_dist = self.head_dist(x)
        return out