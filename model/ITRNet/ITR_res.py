import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

from fvcore.nn import FlopCountAnalysis, parameter_count
import time


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c').contiguous()

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w).contiguous()


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class SAGBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(SAGBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        
    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt



class FrequencyMixer(nn.Module):
    def __init__(self, dim, mix_kernel_size, bias):
        super(FrequencyMixer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, mix_kernel_size, padding=mix_kernel_size//2, stride=1, bias=bias),
            nn.PReLU(),
            nn.BatchNorm2d(dim*2)
        )
        
    def forward(self, x):
        res = x.clone()
        channel_fft = torch.fft.fft2(x, norm='forward', dim=1)
        cat = torch.cat((channel_fft.real, channel_fft.imag), dim=1)
        cat = self.conv1(cat)
        real, imag = torch.chunk(cat, 2, dim=1)
        out = torch.fft.ifft2(torch.complex(real, imag), norm='forward', dim=1)
        return torch.abs(out) + res



class SpatialMixer(nn.Module):
    def __init__(self, dim, mix_kernel_size, bias):
        super(SpatialMixer, self).__init__()

        self.dense_depth_1 = nn.Conv2d(dim, dim, kernel_size=mix_kernel_size, stride=1, padding=mix_kernel_size//2, groups=dim, bias=bias)
        
        self.dense_point_1 = nn.Conv2d(dim, dim,1)

        self.dense_depth_2 = nn.Conv2d(dim, dim, kernel_size=mix_kernel_size, stride=1, padding=mix_kernel_size//2, groups=dim, bias=bias)
        
        self.dense_point_2 = nn.Conv2d(dim, dim,1)

    def forward(self, x):
        mixer_step1 = F.gelu(self.dense_depth_1(x))+x 
        mixer_step1 = x + F.gelu(self.dense_point_1(mixer_step1))

        mixer_step2 = F.gelu(self.dense_depth_2(mixer_step1))+x 
        mixer_step2 = x + F.gelu(self.dense_point_2(mixer_step2))

        return mixer_step2




class DDMBlock(nn.Module):
    def __init__(self, dim, mix_kernel_size, ffn_expansion_factor, bias, LayerNorm_type):
        super(DDMBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fremixer = FrequencyMixer(dim, mix_kernel_size, bias)
        self.spamixer = SpatialMixer(dim, mix_kernel_size, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x_src = x
        z0 = self.norm1(x)
        z1 = x + self.fremixer(z0)
        x = x + self.spamixer(z1)
        x = x_src + self.ffn(self.norm2(x))
    
        return x
  


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



class ITRNet_D(nn.Module):
    def __init__(self, 
        inp_channels=2, 
        out_channels=2, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        mix_kernel_size = [1,3,5,7],
        ffn_expansion_factor = 2.3,
        bias = False,
        LayerNorm_type = 'WithBias',  
        dual_pixel_task = True        
    ):
        super(ITRNet_D, self).__init__()

        self.embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)


        self.prompt1 = SAGBlock(prompt_dim=96,prompt_len=5,prompt_size = 64,lin_dim = 96)
        self.prompt2 = SAGBlock(prompt_dim=192,prompt_len=5,prompt_size = 32,lin_dim = 192)
        self.prompt3 = SAGBlock(prompt_dim=384,prompt_len=5,prompt_size = 16,lin_dim = 384)
    
        self.noise_level3 = DDMBlock(dim=int(dim*2*3) + 480, mix_kernel_size=mix_kernel_size[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2*3)+480,int(dim*2**3),kernel_size=1,bias=bias)

        self.noise_level2 = DDMBlock(dim=int(dim*2*2) + 192, mix_kernel_size=mix_kernel_size[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2*2) + 192,int(dim*2**2),kernel_size=1,bias=bias)

        self.noise_level1 = DDMBlock(dim=int(dim*2) + 96, mix_kernel_size=mix_kernel_size[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2) + 96,int(dim*2),kernel_size=1,bias=bias)


        self.encoder_level1 = nn.Sequential(*[DDMBlock(dim=dim, mix_kernel_size=mix_kernel_size[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) 
        self.encoder_level2 = nn.Sequential(*[DDMBlock(dim=int(dim*2**1), mix_kernel_size=mix_kernel_size[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) 
        self.encoder_level3 = nn.Sequential(*[DDMBlock(dim=int(dim*2**2), mix_kernel_size=mix_kernel_size[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) 
        self.latent = nn.Sequential(*[DDMBlock(dim=int(dim*2**3), mix_kernel_size=mix_kernel_size[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) 
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[DDMBlock(dim=int(dim*2**2), mix_kernel_size=mix_kernel_size[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[DDMBlock(dim=int(dim*2**1), mix_kernel_size=mix_kernel_size[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  
        self.decoder_level1 = nn.Sequential(*[DDMBlock(dim=int(dim*2**1), mix_kernel_size=mix_kernel_size[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
                  
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.dual_pixel_task = dual_pixel_task

    def forward(self, inp_img):
        
        inp_enc_level1 = self.embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)      

        latent = self.latent(inp_enc_level4) 
        
        dec3_param = self.prompt3(latent)
        latent = torch.cat([latent, dec3_param], 1)
        latent = self.noise_level3(latent)
        latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        dec2_param = self.prompt2(out_dec_level3)
        out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
        out_dec_level3 = self.noise_level2(out_dec_level3)
        out_dec_level3 = self.reduce_noise_level2(out_dec_level3)


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        dec1_param = self.prompt1(out_dec_level2)
        out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
        out_dec_level2 = self.noise_level1(out_dec_level2)
        out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        
        if self.dual_pixel_task:
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,-2:,:,:]

        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1
    



if __name__ == '__main__':
    use_transformer = True

    x1 = torch.randn(2, 2, 64, 1920) 

    model = ITRNet_D().cuda().eval()
