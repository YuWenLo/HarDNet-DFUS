import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

# https://github.com/yan-hao-tian/lawin/blob/main/lawin_head.py

class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding: github.com/SwinTransformer/
    """
    def __init__(self, proj_type='conv', patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj_type = proj_type
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if proj_type == 'conv':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif proj_type == 'pool':
            self.proj = nn.ModuleList([nn.MaxPool2d(kernel_size=patch_size, stride=patch_size), nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)])
        else:
            raise NotImplementedError(f'{proj_type} is not currently supported.')
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        
        if self.proj_type == 'conv': 
            x = self.proj(x)  # B C Wh Ww
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x

class LawinAttn(NonLocal2d):
    def __init__(self, *arg, head=1, patch_size=None, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size
        
        if self.head!=1:
            self.position_mixing = nn.ModuleList([nn.Linear(patch_size*patch_size, patch_size*patch_size) for _ in range(self.head)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, query, context):
        # x: [N, C, H, W]
        
        n = context.size(0)
        n, c, h, w = context.shape
        
        if self.head!=1:
            context = context.reshape(n, c, -1)
            context_mlp = []
            for hd in range(self.head):
                context_crt = context[:, (c//self.head)*(hd):(c//self.head)*(hd+1), :]
                context_mlp.append(self.position_mixing[hd](context_crt))

            context_mlp = torch.cat(context_mlp, dim=1)
            context = context+context_mlp
            context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)


        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)

        return output

class LawinHead2(nn.Module):
    def __init__(self, embed_dim=768, use_scale=True, reduction=2, in_channels=[140, 420, 720, 1200], num_classes=1, norm_cfg = dict(type='BN', requires_grad=True), **kwargs):
        super(LawinHead2, self).__init__()
        self.norm_cfg = norm_cfg
        self.lawin_8 = LawinAttn(in_channels=512, reduction=reduction ,use_scale=use_scale, norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=64, patch_size=8)
        self.lawin_4 = LawinAttn(in_channels=512, reduction=reduction ,use_scale=use_scale, norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=16, patch_size=8)
        self.lawin_2 = LawinAttn(in_channels=512, reduction=reduction ,use_scale=use_scale, norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=4, patch_size=8)

        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 512, 1))
        self.linear_c4 = MLP(input_dim=in_channels[-1], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=in_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=in_channels[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=in_channels[0], embed_dim=48)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim*3,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )
        
        self.short_path = ConvModule(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )
        
        self.cat = ConvModule(
            in_channels=512*5,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.low_level_fuse = ConvModule(
            in_channels=560,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )
         
        self.ds_8 = PatchEmbed(proj_type='conv', patch_size=8, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_4 = PatchEmbed(proj_type='conv', patch_size=4, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_2 = PatchEmbed(proj_type='conv', patch_size=2, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)

        self.class_seg = nn.Conv2d(512, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def get_context(self, x, patch_size):
        n, _, h, w = x.shape
        context = []
        for i, r in enumerate([8, 4, 2]):
            _context = F.unfold(x, kernel_size=patch_size*r, stride=patch_size, padding=int((r-1)/2*patch_size))
            _context = rearrange(_context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size*r, pw=patch_size*r, nh=h//patch_size, nw=w//patch_size)
            context.append(getattr(self, f'ds_{r}')(_context))

        return context
    
    def forward(self, c1, c2, c3, c4):
        ############### MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # (n, c, 32, 32)
        _c4 = F.interpolate(_c4, size=c2.size()[2:], mode='bilinear')

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear')

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #(n, c, 128, 128)
        n, _, h, w = _c.shape

        ############### Lawin attention spatial pyramid pooling ###########
        patch_size = 8
        context = self.get_context(_c, patch_size)
        query = F.unfold(_c, kernel_size=patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)
        # sth went wrong if size = 512, query size != stride 8
        output = []
        output.append(self.short_path(_c))

        _c_pool = self.image_pool(_c)
        _c_pool =  F.interpolate(_c_pool, size=(h, w), mode='bilinear')
        output.append(_c_pool)

        for i, r in enumerate([8, 4, 2]):
            _output = getattr(self, f'lawin_{r}')(query, context[i])
            _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)
            output.append(_output)
        #print([x.size() for x in output])
        output = torch.cat(output, dim=1)
        output = self.cat(output)
        output = F.interpolate(output, size=c1.size()[2:], mode='bilinear', align_corners=False)

        ############### Low-level feature enhancement ###########
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))
        output = self.class_seg(output)

        return output, _c, _c1

class LawinHead5(nn.Module):
    def __init__(self, embed_dim=768, use_scale=True, reduction=2, in_channels=[140, 420, 720, 1200], num_classes=1, norm_cfg = dict(type='BN', requires_grad=True), **kwargs):
        super(LawinHead5, self).__init__()
        self.norm_cfg = norm_cfg
        self.lawin_8 = LawinAttn(in_channels=512, reduction=reduction ,use_scale=use_scale, norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=64, patch_size=8)
        self.lawin_4 = LawinAttn(in_channels=512, reduction=reduction ,use_scale=use_scale, norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=16, patch_size=8)
        self.lawin_2 = LawinAttn(in_channels=512, reduction=reduction ,use_scale=use_scale, norm_cfg=self.norm_cfg, mode='embedded_gaussian', head=4, patch_size=8)

        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 512, 1))
        self.linear_c4 = MLP(input_dim=in_channels[-1], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=in_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=in_channels[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=in_channels[0], embed_dim=48)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim*3,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )
        
        self.short_path = ConvModule(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )
        
        self.cat = ConvModule(
            in_channels=512*5,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.low_level_fuse = ConvModule(
            in_channels=560,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )
         
        self.ds_8 = PatchEmbed(proj_type='conv', patch_size=8, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_4 = PatchEmbed(proj_type='conv', patch_size=4, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)
        self.ds_2 = PatchEmbed(proj_type='conv', patch_size=2, in_chans=512, embed_dim=512, norm_layer=nn.LayerNorm)

        self.class_seg = nn.Conv2d(512, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def get_context(self, x, patch_size):
        n, _, h, w = x.shape
        context = []
        for i, r in enumerate([8, 4, 2]):
            _context = F.unfold(x, kernel_size=patch_size*r, stride=patch_size, padding=int((r-1)/2*patch_size))
            _context = rearrange(_context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size*r, pw=patch_size*r, nh=h//patch_size, nw=w//patch_size)
            context.append(getattr(self, f'ds_{r}')(_context))

        return context
    
    def forward(self, c1, c2, c3, c4):
        ############### MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # (n, c, 32, 32)
        _c4 = F.interpolate(_c4, size=c2.size()[2:], mode='bilinear')

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear')

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #(n, c, 128, 128)
        n, _, h, w = _c.shape

        ############### Lawin attention spatial pyramid pooling ###########
        patch_size = 8
        context = self.get_context(_c, patch_size)
        query = F.unfold(_c, kernel_size=patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)
        # sth went wrong if size = 512, query size != stride 8
        output = []
        output.append(self.short_path(_c))

        _c_pool = self.image_pool(_c)
        _c_pool =  F.interpolate(_c_pool, size=(h, w), mode='bilinear')
        output.append(_c_pool)

        for i, r in enumerate([8, 4, 2]):
            _output = getattr(self, f'lawin_{r}')(query, context[i])
            _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)
            output.append(_output)
        #print([x.size() for x in output])
        output = torch.cat(output, dim=1)
        output = self.cat(output)
        output = F.interpolate(output, size=c1.size()[2:], mode='bilinear', align_corners=False)

        ############### Low-level feature enhancement ###########
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))
        output = self.class_seg(output)

        return output, _c, _c4, _c1