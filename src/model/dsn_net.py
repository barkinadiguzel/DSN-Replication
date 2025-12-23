import torch
import torch.nn as nn
from backbone.backbone_block import ConvBlock
from companions.companion_head import CompanionHead

class DSNNet(nn.Module):
    def __init__(self, num_classes=10, num_hidden_layers=3):
        super().__init__()
        self.backbone_layers = nn.ModuleList()
        self.companion_heads = nn.ModuleList()
        in_channels = 3
        for i in range(num_hidden_layers):
            out_channels = 16*(2**i)
            self.backbone_layers.append(ConvBlock(in_channels, out_channels))
            self.companion_heads.append(CompanionHead(out_channels, num_classes))
            in_channels = out_channels

        self.output_head = CompanionHead(out_channels, num_classes)

    def forward(self, x):
        hidden_outputs = []
        for i, layer in enumerate(self.backbone_layers):
            x = layer(x)
            hidden_outputs.append(self.companion_heads[i](x))
        output = self.output_head(x)
        return output, hidden_outputs
