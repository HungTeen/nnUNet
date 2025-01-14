import torch
import torch.nn as nn
import torch.nn.functional as F

from pangteen.network.km_unet.block import KANBlock, ConvLayer, PatchEmbed, D_ConvLayer, SS2D, EMA, CBAM


class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        self.cbam = CBAM(channel=16)
        self.cbam1 = CBAM(channel=32)
        self.cbam2 = CBAM(channel=128)

    # put shape: torch.Size([8, 3, 256, 256])
    # After Stage 1 (encoder1) shape: torch.Size([8, 16, 128, 128])
    # After Stage 2 (encoder2) shape: torch.Size([8, 32, 64, 64])
    # After Stage 3 (encoder3) shape: torch.Size([8, 128, 32, 32])
    # After Stage 4 (patch_embed3) shape: torch.Size([8, 256, 160]), H: 16, W: 16
    # After norm3 and reshape shape: torch.Size([8, 160, 16, 16])
    # After Bottleneck (patch_embed4) shape: torch.Size([8, 64, 256]), H: 8, W: 8
    # After norm4 and reshape shape: torch.Size([8, 256, 8, 8])
    # After decoder1 shape: torch.Size([8, 160, 16, 16])
    # After add t4 shape: torch.Size([8, 160, 16, 16])
    # After dblock1 shape: torch.Size([8, 256, 160])
    # After dnorm3 and reshape shape: torch.Size([8, 160, 16, 16])
    # After decoder2 shape: torch.Size([8, 128, 32, 32])
    # After add t3 shape: torch.Size([8, 128, 32, 32])
    # After dblock2 shape: torch.Size([8, 1024, 128])
    # After dnorm4 and reshape shape: torch.Size([8, 128, 32, 32])
    # After decoder3 shape: torch.Size([8, 32, 64, 64])
    # After add t2 shape: torch.Size([8, 32, 64, 64])
    # After decoder4 shape: torch.Size([8, 16, 128, 128])
    # After add t1 shape: torch.Size([8, 16, 128, 128])
    # After decoder5 shape: torch.Size([8, 16, 256, 256])

    def forward(self, x):

        print(f"Input shape: {x.shape}")
        B = x.shape[0]

        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        # print(f"After Stage 1 (encoder1) shape: {out.shape}")
        t1 = out
        t1 = self.cbam(t1)

        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        # print(f"After Stage 2 (encoder2) shape: {out.shape}")
        t2 = out
        t2 = self.cbam1(t2)

        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        # print(f"After Stage 3 (encoder3) shape: {out.shape}")
        t3 = out
        t3 = self.cbam2(t3)

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        # print(f"After Stage 4 (patch_embed3) shape: {out.shape}, H: {H}, W: {W}")
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After norm3 and reshape shape: {out.shape}")
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        print(f"After Bottleneck (patch_embed4) shape: {out.shape}, H: {H}, W: {W}")
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After norm4 and reshape shape: {out.shape}")

        ### Decoder
        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder1 shape: {out.shape}")
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        # print(f"After add t4 shape: {out.shape}")

        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        # print(f"After dblock1 shape: {out.shape}")

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After dnorm3 and reshape shape: {out.shape}")
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder2 shape: {out.shape}")
        out = torch.add(out, t3)
        # print(f"After add t3 shape: {out.shape}")
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        # print(f"After dblock2 shape: {out.shape}")

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"After dnorm4 and reshape shape: {out.shape}")

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder3 shape: {out.shape}")
        out = torch.add(out, t2)
        # print(f"After add t2 shape: {out.shape}")
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder4 shape: {out.shape}")
        out = torch.add(out, t1)
        # print(f"After add t1 shape: {out.shape}")
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        # print(f"After decoder5 shape: {out.shape}")

        return self.final(out)


#     def forward(self, x):
# #        x = self.cbam(x)

#         print(f"Input shape111111111111111111111111111: {x.shape}")
#         B = x.shape[0]
#         ### Encoder
#         ### Conv Stage

#         ### Stage 1
#         out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
#         t1 = out
#         ### Stage 2
#         out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
#         t2 = out
#         ### Stage 3
#         out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
#         t3 = out

#         ### Tokenized KAN Stage
#         ### Stage 4

#         out, H, W = self.patch_embed3(out)
#         for i, blk in enumerate(self.block1):
#             out = blk(out, H, W)
#         out = self.norm3(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         t4 = out

#         ### Bottleneck

#         out, H, W= self.patch_embed4(out)
#         for i, blk in enumerate(self.block2):
#             out = blk(out, H, W)
#         out = self.norm4(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         ### Stage 4
#         out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode ='bilinear'))

#         out = torch.add(out, t4)
#         _, _, H, W = out.shape
#         out = out.flatten(2).transpose(1,2)
#         for i, blk in enumerate(self.dblock1):
#             out = blk(out, H, W)

#         ### Stage 3
#         out = self.dnorm3(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
#         out = torch.add(out,t3)
#         _,_,H,W = out.shape
#         out = out.flatten(2).transpose(1,2)

#         for i, blk in enumerate(self.dblock2):
#             out = blk(out, H, W)

#         out = self.dnorm4(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
#         out = torch.add(out,t2)
#         out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
#         out = torch.add(out,t1)
#         out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

#         return self.final(out)

class KM_UNet(nn.Module):
    """
    Reference: https://github.com/2760613195/KM_UNet
    """

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        self.cbam = CBAM(channel=16)
        self.cbam1 = CBAM(channel=32)
        self.cbam2 = CBAM(channel=128)
        # SS2D模块
        self.ss2d_1 = SS2D(d_model=16)  # Stage 1后
        self.ss2d_2 = SS2D(d_model=32)  # Stage 2后
        self.ss2d_3 = SS2D(d_model=128)  # Stage 3后
        self.ss2d_decoder1 = SS2D(d_model=160)  # Decoder1后
        self.ss2d_decoder2 = SS2D(d_model=128)  # Decoder2后
        self.ss2d_decoder3 = SS2D(d_model=32)  # Decoder3后
        self.ss2d_decoder4 = SS2D(d_model=16)  # Decoder4后
        # EMA注意力机制
        self.ema1 = EMA(channels=16)
        self.ema2 = EMA(channels=32)
        self.ema3 = EMA(channels=128)
        self.ema_decoder1 = EMA(channels=160)
        self.ema_decoder2 = EMA(channels=128)
        self.ema_decoder3 = EMA(channels=32)
        self.ema_decoder4 = EMA(channels=16)

    def forward(self, x):

        print(f"Input shape: {x.shape}")
        B = x.shape[0]

        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))  # 输出通道数为16
        t1 = out
        # t1 = self.cbam(t1)
        t1 = t1.permute(0, 2, 3, 1).contiguous()  # 从BCHW -> BHWC
        t1 = self.ss2d_1(t1)  # SS2D的d_model设置为16
        t1 = t1.permute(0, 3, 1, 2).contiguous()  # 从BHWC -> BCHW
        t1 = self.ema1(t1)  # Apply EMA

        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))  # 输出通道数为32
        t2 = out
        # t2 = self.cbam1(t2)
        t2 = t2.permute(0, 2, 3, 1).contiguous()  # 从BCHW -> BHWC
        t2 = self.ss2d_2(t2)  # SS2D的d_model设置为32
        t2 = t2.permute(0, 3, 1, 2).contiguous()  # 从BHWC -> BCHW
        t2 = self.ema2(t2)  # Apply EMA

        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))  # 输出通道数为128
        t3 = out
        # t3 = self.cbam2(t3)
        t3 = t3.permute(0, 2, 3, 1).contiguous()  # 从BCHW -> BHWC
        t3 = self.ss2d_3(t3)  # SS2D的d_model设置为128
        t3 = t3.permute(0, 3, 1, 2).contiguous()  # 从BHWC -> BCHW
        t3 = self.ema3(t3)  # Apply EMA

        ### Tokenized KAN Stage
        ### Stage 4
        out, H, W = self.patch_embed3(out)  # 输出通道数为256
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck
        out, H, W = self.patch_embed4(out)  # 输出通道数为64
        print(f"After Bottleneck (patch_embed4) shape: {out.shape}, H: {H}, W: {W}")
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder
        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))  # 输出通道数为160
        out = torch.add(out, t4)
        out = out.permute(0, 2, 3, 1).contiguous()  # 从BCHW -> BHWC
        out = self.ss2d_decoder1(out)  # SS2D的d_model设置为160
        out = out.permute(0, 3, 1, 2).contiguous()  # 从BHWC -> BCHW
        out = self.ema_decoder1(out)  # Apply EMA

        ### Stage 3
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))  # 输出通道数为128
        out = torch.add(out, t3)
        out = out.permute(0, 2, 3, 1).contiguous()  # 从BCHW -> BHWC
        out = self.ss2d_decoder2(out)  # SS2D的d_model设置为128
        out = out.permute(0, 3, 1, 2).contiguous()  # 从BHWC -> BCHW
        out = self.ema_decoder2(out)  # Apply EMA

        ### Stage 2
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))  # 输出通道数为32
        out = torch.add(out, t2)
        out = out.permute(0, 2, 3, 1).contiguous()  # 从BCHW -> BHWC
        out = self.ss2d_decoder3(out)  # SS2D的d_model设置为32
        out = out.permute(0, 3, 1, 2).contiguous()  # 从BHWC -> BCHW
        out = self.ema_decoder3(out)  # Apply EMA

        ### Stage 1
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))  # 输出通道数为16
        out = torch.add(out, t1)
        out = out.permute(0, 2, 3, 1).contiguous()  # 从BCHW -> BHWC
        out = self.ss2d_decoder4(out)  # SS2D的d_model设置为16
        out = out.permute(0, 3, 1, 2).contiguous()  # 从BHWC -> BCHW
        out = self.ema_decoder4(out)  # Apply EMA

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))  # 输出通道数为16

        return self.final(out)


class UKANPP(nn.Module):  # UKANPP stands for U-Net++ with KAN
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        # Encoding layers (U-Net++)
        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        # Normalization layers
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        # D-Block normalization layers
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        # DropPath initialization for stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Tokenized KAN layers
        self.block1 = nn.ModuleList(
            [KANBlock(dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])
        self.block2 = nn.ModuleList(
            [KANBlock(dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        # Decoder blocks (for upsampling and U-Net++ style connections)
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  # Upsample and connect with encoder 3 output
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  # Upsample and connect with encoder 2 output
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)  # Upsample and connect with encoder 1 output
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        # Patch Embedding layers for KAN integration
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        # U-Net++ dense connections
        self.plusplus_conv2_1 = ConvLayer(embed_dims[1], embed_dims[0])
        self.plusplus_conv3_2 = ConvLayer(embed_dims[2], embed_dims[1])

        # Final output layer
        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]

        # Encoder stages with dense connections
        out1 = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))  # Encoder 1
        out2 = F.relu(F.max_pool2d(self.encoder2(out1), 2, 2))  # Encoder 2
        out3 = F.relu(F.max_pool2d(self.encoder3(out2), 2, 2))  # Encoder 3

        # Patch embedding and KAN blocks (tokenized KAN processing)
        out, H, W = self.patch_embed3(out3)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # U-Net++ style decoder with dense connections
        upsampled = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear'))
        upsampled = torch.add(upsampled, out3)  # Add encoder3 output

        upsampled = F.relu(F.interpolate(self.decoder2(upsampled), scale_factor=2, mode='bilinear'))
        upsampled = torch.add(upsampled, out2)  # Add encoder2 output

        # Add dense connections in U-Net++ manner
        out2_plus = F.relu(self.plusplus_conv2_1(out2))
        upsampled = torch.add(upsampled, out2_plus)

        upsampled = F.relu(F.interpolate(self.decoder3(upsampled), scale_factor=2, mode='bilinear'))
        upsampled = torch.add(upsampled, out1)  # Add encoder1 output

        out1_plus = F.relu(self.plusplus_conv3_2(out1))
        upsampled = torch.add(upsampled, out1_plus)

        # Final upsampling layers
        upsampled = F.relu(F.interpolate(self.decoder4(upsampled), scale_factor=2, mode='bilinear'))
        upsampled = F.relu(F.interpolate(self.decoder5(upsampled), scale_factor=2, mode='bilinear'))

        return self.final(upsampled)
