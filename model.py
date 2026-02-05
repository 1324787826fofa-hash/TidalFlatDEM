class UNet_improved(nn.Module):



    def __init__(self, n_channels=7, n_classes=1):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # PSA modules on skip connections
        self.psa1 = PSA(64)
        self.psa2 = PSA(128)
        self.psa3 = PSA(256)
        self.psa4 = PSA(512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # PSA-enhanced skip connections
        x1 = self.psa1(x1)
        x2 = self.psa2(x2)
        x3 = self.psa3(x3)
        x4 = self.psa4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)
