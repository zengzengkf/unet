import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
from EMA import EMA

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x
# class UpsampleConv(nn.Module):
#     def __init__(self,
#                 in_features,
#                 out_features,
#                 kernel_size=(3, 3),
#                 padding=(1, 1),
#                 norm_type=None,
#                 activation=False,
#                 scale=(2, 2),
#                 conv='conv') -> None:
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=scale,
#                               mode='bilinear',
#                               align_corners=True)
#         if conv == 'conv':
#             self.conv = conv_block(in_features=in_features,
#                                     out_features=out_features,
#                                     kernel_size=(1, 1),
#                                     padding=(0, 0),
#                                     norm_type=norm_type,
#                                     activation=activation)
#         elif conv == 'depthwise':
#             self.conv = depthwise_conv_block(in_features=in_features,
#                                     out_features=out_features,
#                                     kernel_size=kernel_size,
#                                     padding=padding,
#                                     norm_type=norm_type,
#                                     activation=activation)
#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#         return x
def normal_init(module, mean=0, std=1.0, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h],indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h],indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)



class PoolEmbedding(nn.Module):
    def __init__(self,
                pooling,
                patch,
                ) -> None:
        super().__init__()
        self.projection = pooling(output_size=(patch[0], patch[1]))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x

class ScaleDotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3)
        return x123

class depthwise_conv_block(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=None,
                norm_type='bn',
                activation=True,
                use_bias=True,
                pointwise=False,
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features,
                                        out_features,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        dilation=(1, 1),
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class depthwise_projection(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 patch,
                 groups,
                 kernel_size=(1, 1),
                 padding=(0, 0),
                 norm_type=None,
                 activation=False,
                 pointwise=False) -> None:
        super().__init__()
        self.patch = patch
        self.proj = depthwise_conv_block(in_features=in_features,
                                         out_features=out_features,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         groups=groups,
                                         pointwise=pointwise,
                                         norm_type=norm_type,
                                         activation=activation)

    def forward(self, x):
        H, W = self.patch
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=H, W=W)
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x

# 假设784*32 784*480 784*480
class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1,patch=(12,16)) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          patch=patch,
                                          groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          patch=patch,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          patch=patch,
                                          groups=in_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               patch=patch,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att

class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4,patch=(12,16)) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          patch=patch,
                                          groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          patch=patch,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          patch=patch,
                                          groups=out_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               patch=patch,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        x = self.projection(att)
        return x

class CCSABlock(nn.Module):
    def __init__(self,
                 features,
                 channel_head,
                 spatial_head,
                 spatial_att=True,
                 channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.c_attention = nn.ModuleList([ChannelAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            ) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.s_attention = nn.ModuleList([SpatialAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            )
                for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)

class DCA(nn.Module):
    def __init__(self,
                 features,
                 dropout_rate,
                 strides=(8, 4, 2, 1),
                 patch=(12,16),
                 channel_att=True,
                 spatial_att=True,
                 n=1,
                 channel_head=(1, 1, 1, 1),
                 spatial_head=(4, 4, 4, 4),
                 ):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
            pooling=nn.AdaptiveAvgPool2d,
            patch=patch,
        )
            for _ in features])
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                           out_features=feature,
                                                           patch=patch,
                                                           kernel_size=(1, 1),
                                                           padding=(0, 0),
                                                           groups=feature
                                                           )
                                      for feature in features])

        self.attention = nn.ModuleList([
            CCSABlock(features=features,
                      channel_head=channel_head,
                      spatial_head=spatial_head,
                      channel_att=channel_att,
                      spatial_att=spatial_att)
            for _ in range(n)])

        self.upconvs = nn.ModuleList([DySample(in_channels=feature,scale=stride)
                                      for feature, stride in zip(features, strides)])
        self.bn_relu = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=feature,out_channels=feature,kernel_size=(1,1)),
            nn.BatchNorm2d(feature),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
        )
            for feature in features])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out,)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch[0], W=self.patch[1])



class DCA1(nn.Module):
    def __init__(self,num=4,features=(32, 64, 128, 256)):
        super().__init__()
        self.ema1 = EMA(features[0])
        self.ema2 = EMA(features[1])
        self.ema3 = EMA(features[2])
        self.ema4 = EMA(features[3])
        dropout_rate = (0.1,0.2,0.3,0.4)
        self.groups = nn.ModuleList()
        for i in range(num):
            self.groups.append(DCA(features=features,dropout_rate=dropout_rate[i]))

    def forward(self, raw):
        x1,x2,x3,x4 = raw[0], raw[1], raw[2], raw[3]

        for group in self.groups:
            x1,x2,x3,x4 = group((x1,x2,x3,x4))
        x1 = self.ema1(x1+raw[0])
        x2 = self.ema2(x2+raw[1])
        x3 = self.ema3(x3+raw[2])
        x4 = self.ema4(x4+raw[3])

        return x1,x2,x3,x4


if __name__ == '__main__':
    features = [32, 64, 128, 256]
    dca_model = DCA(features=features,dropout_rate=0.1)
    input1 = torch.randn(1, features[0], 96, 128)
    input2 = torch.randn(1, features[1], 48 ,64)
    input3 = torch.randn(1, features[2], 24, 32)
    input4 = torch.randn(1, features[3], 12, 16)

    print("Input Shapes:")
    print("input1:", input1.shape)
    print("input2:", input2.shape)
    print("input3:", input3.shape)
    print("input4:", input4.shape)

    output1, output2, output3, output4 = dca_model((input1, input2, input3, input4))

    print("\nOutput Shapes:")
    print("output1:", output1.shape)
    print("output2:", output2.shape)
    print("output3:", output3.shape)
    print("output4:", output4.shape)

    dca1_model = DCA1(features=features)
    output1,output2,output3,output4 = dca1_model((input1, input2, input3, input4))
    print("\nOutput Shapes:")
    print("output1:", output1.shape)
    print("output2:", output2.shape)
    print("output3:", output3.shape)
    print("output4:", output4.shape)
