
# resnet unet
from components import *

class UNet(nn.Module):
    def __init__(self,base_channel=32):
        super(UNet, self).__init__()
        # 编码器
        self.enc0 = nn.Sequential(
            ResnetBlock(1, base_channel,2),
            HWDownsampling(base_channel, base_channel)
        )

        self.enc1 = nn.Sequential(
            ResnetBlock(base_channel, base_channel*2,2),
            HWDownsampling(base_channel*2, base_channel*2)
        )

        self.enc2 = nn.Sequential(
           ResnetBlock(base_channel*2, base_channel*4,4,dropout_rate=0.05),
            HWDownsampling(base_channel*4, base_channel*4)
        )

        self.enc3 = nn.Sequential(
            ResnetBlock(base_channel*4, base_channel*8,4,dropout_rate=0.1),
            HWDownsampling(base_channel*8, base_channel*8)
        )

        self.enc4 = nn.Sequential(
            ResnetBlock(base_channel*8, base_channel*16,6,dropout_rate=0.2),
            HWDownsampling(base_channel*16, base_channel*16)
        )

        # 瓶颈层
        self.bottleneck = ResnetBlock(base_channel*16, base_channel*16,6,dropout_rate=0.4)
        # self.dca = DCA(features=(base_channel, base_channel*2, base_channel*4, base_channel*8))
        # self.ca4 = CoordAtt(inp=base_channel * 16, oup=base_channel * 16)
        # self.ca3 = CoordAtt(inp=base_channel * 8, oup=base_channel * 8)
        # self.ca2 = CoordAtt(inp=base_channel * 4, oup=base_channel * 4)
        # self.ca1 = CoordAtt(inp=base_channel * 2, oup=base_channel * 2)

        # 解码器上采样层
        self.up4 = DySample(in_channels=base_channel*16)
        self.conv4 = ResnetBlock(base_channel*16, base_channel*8,3,dropout_rate=0.2)
        self.dec4 = ResnetBlock(base_channel*16, base_channel*8,3,dropout_rate=0.2)

        self.up3 = DySample(in_channels=base_channel * 8)
        self.conv3 = ResnetBlock(base_channel * 8, base_channel*4,2,dropout_rate=0.1)
        self.dec3 = ResnetBlock(base_channel * 8, base_channel * 4,2,dropout_rate=0.1)


        self.up2 = DySample(in_channels=base_channel * 4)
        self.conv2 = ResnetBlock(base_channel * 4, base_channel*2,2,dropout_rate=0.05)
        self.dec2 = ResnetBlock(base_channel * 4, base_channel*2,2,dropout_rate=0.05)


        self.up1 = DySample(in_channels=base_channel * 2)
        self.conv1 = ResnetBlock(base_channel * 2, base_channel,1)
        self.dec1 = ResnetBlock(base_channel*2, base_channel,1)
        # 最终输出层
        self.up0 = DySample(in_channels=base_channel)
        self.final = nn.Conv2d(base_channel, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # 编码器
        p1 = self.enc0(x)
        p2 = self.enc1(p1)
        p3 = self.enc2(p2)
        p4 = self.enc3(p3)
        b = self.enc4(p4)

        # 瓶颈层
        b = self.bottleneck(b)
        # p1,p2,p3,p4 = self.dca((p1,p2,p3,p4))
        #
        #
        # # 解码器
        u4 = self.conv4(self.up4(b))
        u4 = torch.cat([u4, p4], dim=1)
        # u4 = self.ca4(u4)
        d4 = self.dec4(u4)
        #
        u3 = self.conv3(self.up3(d4))
        u3 = torch.cat([u3, p3], dim=1)
        # u3 = self.ca3(u3)
        d3 = self.dec3(u3)
        #
        u2 = self.conv2(self.up2(d3))
        u2 = torch.cat([u2, p2], dim=1)
        # u2 = self.ca2(u2)
        d2 = self.dec2(u2)
        #
        u1 = self.conv1(self.up1(d2))
        u1 = torch.cat([u1, p1], dim=1)
        # u1 = self.ca1(u1)
        d1 = self.dec1(u1)
        #
        out = self.final(self.up0(d1))
        return out

if __name__ == '__main__':

    # 实例化模型
    model = UNet()
    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数数量为: {total_params}")
    # 验证输入输出尺寸
    test_input = torch.randn(1, 1, 192, 256)
    output = model(test_input)
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
    print_parameters(model)