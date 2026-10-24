import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CheckerboardFreeUpsample(nn.Module):
    """Upsamples using Bilinear Interpolation + Conv2d to physically prevent grids."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class LunarUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(LunarUNet, self).__init__()

        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Bottleneck
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Decoder (Upsampling - now using checkerboard-free Bilinear Blocks)
        self.up1 = CheckerboardFreeUpsample(512, 256)
        self.conv1 = DoubleConv(512, 256) # 512 due to skip connections
        
        self.up2 = CheckerboardFreeUpsample(256, 128)
        self.conv2 = DoubleConv(256, 128)
        
        self.up3 = CheckerboardFreeUpsample(128, 64)
        self.conv3 = DoubleConv(128, 64)

        # Final Convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        y = self.up1(x4)
        y = torch.cat([x3, y], dim=1)
        y = self.conv1(y)
        
        y = self.up2(y)
        y = torch.cat([x2, y], dim=1)
        y = self.conv2(y)
        
        y = self.up3(y)
        y = torch.cat([x1, y], dim=1)
        y = self.conv3(y)
        
        logits = self.outc(y)
        return self.sigmoid(logits)

if __name__ == "__main__":
    mock_input = torch.randn(1, 1, 256, 256)
    model = LunarUNet()
    mock_output = model(mock_input)
    print(f"Model built! Output shape: {mock_output.shape} (Grid-free architecture)")