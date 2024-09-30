# unet_dcr_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCRBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(DCRBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, int(1.5 * in_channels), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(int(1.5 * in_channels))

        self.conv2 = nn.Conv3d(int(2.5 * in_channels), int(2 * in_channels), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(int(2 * in_channels))

        self.conv3 = nn.Conv3d(int(5.5 * in_channels), in_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        conca1 = torch.cat([x, c1], 1)

        c2 = F.relu(self.bn2(self.conv2(conca1)))
        conca2 = torch.cat([x, conca1, c2], 1)

        c3 = F.relu(self.bn3(self.conv3(conca2)))
        return c3 + x

class UNetWithDCR3D(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(UNetWithDCR3D, self).__init__()

        # Encoding Path
        self.CONV1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.DCR1 = DCRBlock3D(base_channels)

        self.CONV2 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.DCR2 = DCRBlock3D(base_channels * 2)

        self.CONV3 = nn.Conv3d(base_channels * 2, base_channels * 3, kernel_size=3, padding=1)
        self.DCR3 = DCRBlock3D(base_channels * 3)

        # Decoding Path
        self.CONV5 = nn.Conv3d(base_channels * 3, base_channels, kernel_size=3, padding=1)
        self.CONV6 = nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.CONV7 = nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1)

        self.final_conv = nn.Conv3d(base_channels * 2, 1, kernel_size=1)

    def forward(self, x):
        # Encoding path
        conv1 = F.relu(self.CONV1(x))
        dcr1 = self.DCR1(conv1)
        m1 = F.max_pool3d(dcr1, 2)
        
        conv2 = F.relu(self.CONV2(m1))
        dcr2 = self.DCR2(conv2)
        m2 = F.max_pool3d(dcr2, 2)
        
        conv3 = F.relu(self.CONV3(m2))
        dcr3 = self.DCR3(conv3)
        m3 = F.max_pool3d(dcr3, 2)

        # Decoding path
        u3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(m3)
        c5 = F.relu(self.CONV5(u3))
        c6 = F.relu(self.CONV5(dcr3))
        con3 = torch.cat([c5, c6], 1)
        
        u2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(con3)
        c3 = F.relu(self.CONV6(u2))
        c4 = F.relu(self.CONV6(dcr2))
        con2 = torch.cat([c3, c4], 1)
        
        u1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(con2)
        c1 = F.relu(self.CONV6(u1))
        c2 = F.relu(self.CONV7(dcr1))
        con = torch.cat([c1, c2], 1)
        
        ddcr1 = self.DCR2(con)
        convfinal = torch.sigmoid(self.final_conv(ddcr1))
        return convfinal

def create_model(in_channels=1, base_channels=32, device='cpu'):
    model = UNetWithDCR3D(in_channels, base_channels).to(device)
    return model

if __name__ == "__main__":
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(in_channels=1, base_channels=32, device=device)
    # input_tensor = torch.rand((batch_size, 1, 64, 64, 64)).to(device)

    # Pass the input tensor through the model
    # output = model(input_tensor)

    # Print the shape of the output tensor
    # print(output.shape)
