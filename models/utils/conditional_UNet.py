from torch import nn
from models.utils.UNet import UNet
from models.utils.diffusion_utils import exists

class UNetConditional(UNet):
    def __init__(
        self,
        img_size,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        num_classes=14,
        **kwargs,
    ):
        super().__init__(img_size, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.time_dim)


    def forward(self, x, time, y=None):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if y is not None:
            t += self.label_emb(y)


        return self.unet_forward(x, t)

