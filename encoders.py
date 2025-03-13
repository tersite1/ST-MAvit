import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x

class MaxViTBlock(nn.Module):
    def __init__(self, dim, window_size, dilation, num_heads):
        super(MaxViTBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, padding=dilation)

    def forward(self, x):
        x = self.attention(x, x, x)[0]
        x = self.conv(x)
        return x

class AnchorAwareAttention(nn.Module):
    def __init__(self, dim, num_heads, anchor_dim):
        super(AnchorAwareAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.fc = nn.Linear(dim, anchor_dim)

    def forward(self, x):
        x = self.attention(x, x, x)[0]
        return self.fc(x)

class STMAViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=256, patch_size=config["patch_embedding"]["patch_size"], 
            in_chans=13, embed_dim=config["patch_embedding"]["embed_dim"]
        )
        self.maxvit_blocks = nn.Sequential(
            *[MaxViTBlock(dim=config["maxvit_blocks"][0]["dim"], 
                          window_size=config["maxvit_blocks"][0]["window_size"], 
                          dilation=config["maxvit_blocks"][0]["dilation"],
                          num_heads=config["maxvit_blocks"][0]["num_heads"]) for _ in range(4)]
        )
        self.anchor_attention = AnchorAwareAttention(
            dim=config["maxvit_blocks"][0]["dim"], 
            num_heads=config["maxvit_blocks"][0]["num_heads"], 
            anchor_dim=512
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.maxvit_blocks(x)
        x = self.anchor_attention(x)
        return x
