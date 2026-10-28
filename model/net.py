
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers import DecoderLayer
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class jtcMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class jtcAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False,
                 vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv=3,27,8,17,64
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)



        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class jtcBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=jtcAttention, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                 depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            comb=comb, vis=vis)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = jtcMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x
class SCMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class SCAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SCBlock(nn.Module):

    def __init__(self ,dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SCAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class SCBlock0(nn.Module):

    def __init__(self ,dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SCAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class Decoder(nn.Module):

    def __init__(
            self,  n_layers, n_head, d_k, d_v,
            d_model, d_inner,  dropout=0.1, device='cuda'):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.device = device

    def forward(self, trg_seq, enc_output, return_attns=False):

        dec_enc_attn_list = []
        dec_output = (trg_seq)  # bs * person, 3 * person + input_frames, dim=128
        layer=0
        for dec_layer in self.layer_stack:
            layer+=1
            dec_output, dec_enc_attn = dec_layer(
                dec_output, enc_output)
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_enc_attn_list
        return dec_output


def body_partition(mydata, index):   # Body Partition
    bn, seq_len, _ = mydata.shape
    mydata = mydata.reshape(bn, seq_len, -1, 32)  # 96, 50, 15, 3
    out = torch.zeros(bn, seq_len, len(index), 32).to(mydata.device)  # x, 12, 3, 35
    for i in range(len(index)):
        temp1 = mydata[:, :, index[i], :].reshape(-1, len(index[i]), 32).transpose(1,2)
       # temp2 = torch.mean(temp1, dim=-1, keepdim=True)
        temp2 = F.avg_pool1d(temp1, kernel_size=5, padding=1)
        temp2 = temp2.transpose(1, 2).reshape(bn, seq_len, -1, 32)
        out[:, :, i, :] = temp2[:, :, 0, :]
    return out

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_length, embed_dim)

    def forward(self, inputs):
        seq_length = inputs.size(1)
        positions = torch.arange(seq_length, device=inputs.device)
        position_embeddings = self.position_embeddings(positions)
        return inputs + position_embeddings
class Tem_ID_Encoder(nn.Module):
    def __init__(self, d_model, dropout=0.1,
                 max_t_len=200, max_a_len=20):
        super(Tem_ID_Encoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        ie = self.build_id_enc(max_a_len)
        self.register_buffer('ie', ie)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        return pe

    def build_id_enc(self, max_len):
        ie = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ie[:, 0::2] = torch.sin(position * div_term)
        ie[:, 1::2] = torch.cos(position * div_term)
        ie = ie.unsqueeze(0)
        return ie

    def get_pos_enc(self, num_a, num_p, num_t, t_offset):
        pe = self.pe[:, t_offset: num_t + t_offset]

        pe = pe.repeat(1, num_a*num_p, 1)

        return pe

    def get_id_enc(self, num_p, num_t, i_offset, id_enc_shuffle):

        ie = self.ie[:, id_enc_shuffle]
        ie = ie.repeat_interleave(num_p*num_t, dim=1)


        return ie

    def forward(self, x, num_a, num_p, num_t, t_offset=0, i_offset=0):

        index = list(np.arange(0, num_p))
        id_enc_shuffle = random.sample(index, num_a)
        pos_enc = self.get_pos_enc(num_a, num_p, num_t, t_offset)
        id_enc = self.get_id_enc(num_p, num_t, i_offset, id_enc_shuffle)
        x = x + pos_enc + id_enc     #  Temporal Encoding + Identity Encoding
        return self.dropout(x)

class CONSFormer(nn.Module):

    def __init__(
            self, input_dim=128, d_model=512, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64, dropout=0.2,
            device='cuda', kernel_size=10, opt=None):

        super().__init__()
        self.kernel_size = opt.kernel_size
        self.device = device
        self.d_model = d_model

        self.conv2d = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=input_dim, kernel_size=(1, opt.kernel_size), stride=(1, 1), bias=False),
                                nn.ReLU(inplace=False))



        self.decoder = Decoder(d_model=d_model, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, device=self.device)


        kernel_size1 = int(kernel_size/2+1)
        if kernel_size%2==0:
            kernel_size2 =  int(kernel_size/2)
        else:
            kernel_size2 =  int(kernel_size/2+1)
        self.mlp = nn.Sequential(nn.Conv1d(in_channels=15*32, out_channels=d_model, kernel_size=kernel_size1,
                                             bias=False),
                                   nn.ReLU(inplace=False),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size2,
                                             bias=False),
                                   nn.ReLU(inplace=False))

        self.proj_inverse = nn.Linear(d_model, 15 * d_model)
        self.proj_inverse1 = nn.Linear(d_model * 15, 15 * 3)
        self.l1=nn.Linear(d_model, d_model*5)
        n_position=1000
        self.embeddings = Tem_ID_Encoder(d_model, dropout=dropout,
                                         max_t_len=n_position, max_a_len=1000)  # temporal encodings + identity encodings



        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == input_dim, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'
        embed_dim_ratio = 128
        drop_rate = 0.
        num_heads = 8
        depth = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        attn_drop_rate = 0.
        dropout = 0.05
        self.drop = nn.Dropout(dropout)
        # jtc
        if opt.train_batch:
            drop_path_rate = 0.05
        else:
            drop_path_rate = 0
        self.block_depth = 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]
        norm_layer = None or partial(nn.LayerNorm, eps=1e-6)
        self.Spatial_patch_to_embedding = nn.Linear(3, 32)
        self.Spatial_pos_embed0 = nn.Parameter(torch.zeros(1, 15, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Tpatial_blocks = nn.ModuleList([
            SCBlock(dim=embed_dim_ratio*15, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(3)])
        self.STpatial_blocks = nn.ModuleList([
            SCBlock0(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(3)])
        self.Spatial_blocks = nn.ModuleList([
            SCBlock(dim=128, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(3)])
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        max_length = 2000  #
        self.position_encoder=LearnablePositionalEncoding(15*32, max_length)
        self.position_encoder1 = LearnablePositionalEncoding(128, max_length)
        self.jtcblocks = nn.ModuleList([
            jtcBlock(
                dim=128, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False,
                changedim=False, currentdim=i + 1, depth=depth)
            for i in range(3)])
        self.fusion_conv = nn.Conv2d(128, 128, kernel_size=1)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(3 * 15, 15 * d_model)



    def jtcpos(self,src):
        b, jn, c = src.shape
        p = nn.Parameter(torch.zeros(1, jn, c))
        return p

    def ntpos(self,src):
        b, jn, c = src.shape
        p = nn.Parameter(torch.zeros(1, jn, c))
        return p

    def forward(self, src,src_gt, n_person):
        '''
        src_seq:  B*N, T, J*3
        '''

        bn,T1,_=src.shape
        bs = int(bn / n_person)
        src1 = src.view(bn *T1, 15, -1)
        src1 = self.Spatial_patch_to_embedding(src1)
        src0 = src1.view(bn, T1, 15 * 32)
        src1 = src1.view(bn, 32, 15, T1)
        src1 = self.conv2d(src1).permute(0, 2, 3, 1).reshape(bs, n_person, 15, -1,128)  # multi-person body parts sequence
        bs,n_person,j,T,_=src1.shape
        src1 = src1.reshape(bn * T, 15, -1)
        src1 += self.Spatial_pos_embed0
      
        src1 = src1.view(bs*T , n_person * 15, 128)
        src1 += self.jtcpos(src1).to(src1.device)
        for i in range(2):
           jtcblock = self.jtcblocks[i]
           src1 = jtcblock(src1)
        src1 = src1.view(bn * 15, T, -1)
       
        src1 = src1.view(bn, T, 15 * 128)

        mpbp_seq =src1.reshape(bs *n_person * 15,T, -1)
        mpbp_seq += self.ntpos(mpbp_seq).to(mpbp_seq.device)
        mpbp_seq = self.pos_drop(mpbp_seq)

        for i in range(2):
            blk = self.Spatial_blocks[i]
            mpbp_seq = blk(mpbp_seq)
        mpbp_seq = mpbp_seq.view(bn, 128, 15, T)
       
        mpbp_seq0=mpbp_seq
        mpbp_seq_gt=mpbp_seq
        mpbp_seq0_gt = mpbp_seq
        src0_gt=src0
        concat_dec_out0 = []
        concat=[]
        concat_dec_out0_gt = []
        concat_gt = []
        concat_gtgt=[]

        for i in range(9):

            enc_out = mpbp_seq.reshape(bs, -1, 128)
            # ======= Transformer Decoder ============
            src_query = src0.transpose(1, 2)[:, :, -self.kernel_size:].clone()
            global_body_query = self.mlp(src_query).reshape(bs, n_person, -1)
            dec_output = self.decoder(global_body_query, enc_out, False)
            dec_output = dec_output.reshape(bn, 1, -1)
            # =======  FC ============
            dec_output = self.l1(dec_output)
            dec_output = dec_output.view(bn, 5, -1)
            dec_out0 = self.proj_inverse(dec_output)
            dec_out0 = dec_out0.view(bs*5, 15*n_person, -1)
            for i in range(1):
                blk = self.jtcblocks[i]
                dec_out0 = blk(dec_out0)
            dec_out0 = dec_out0.view(bs *15 * n_person,5, -1)
            for j in range(1):
                blk = self.Spatial_blocks[j]
                dec_out0 = blk(dec_out0)
            dec_out0 = dec_out0.view(bn, 128, 15, 5)
            concat.append(dec_out0)
            final = torch.cat(concat, dim=3)
            mpbp_seq = torch.cat((mpbp_seq0,final),dim=3)


            src00 = dec_out0.view(bn, 5, -1)
            src00 = self.proj_inverse1(src00)
            src00 = src00.view(bn * 5, 15, -1)
            src00 = self.Spatial_patch_to_embedding(src00)
            src00 = src00.view(bn ,5, 15*32)
            src0=torch.cat((src0,src00),dim=1)
            bn,Cw,J,Tw=mpbp_seq.shape
            src0 = self.position_encoder(src0)
            mpbp_seq = mpbp_seq.view(bn*15, -1,128)
            mpbp_seq = self.position_encoder1(mpbp_seq)
            mpbp_seq = self.pos_drop(mpbp_seq)
            for j in range(1):
                blk = self.Spatial_blocks[j]
                mpbp_seq = blk(mpbp_seq)
            concat_dec_out0.append(dec_out0)
            mpbp_seq = mpbp_seq.view(bn, 128, 15, -1)
        final_dec_out0 = torch.cat(concat_dec_out0, dim=3)
        final_dec_out0 = final_dec_out0.view(bn, 45, -1)
        final_dec_out0 = self.proj_inverse1(final_dec_out0)

        for i in range(9):

            enc_out_gt = mpbp_seq_gt.reshape(bs, -1, 128)
           
            # ======= Transformer Decoder ============
          
            src_query_gt = src0_gt.transpose(1, 2)[:, :, -self.kernel_size:].clone()
            global_body_query_gt = self.mlp(src_query_gt).reshape(bs, n_person, -1)
            dec_output_gt = self.decoder(global_body_query_gt, enc_out_gt, False)
          
            dec_output_gt = dec_output_gt.reshape(bn, 1, -1)

            # =======  FC ============
            dec_output_gt = self.l1(dec_output_gt)

            dec_output_gt = dec_output_gt.view(bn, 5, -1)

            dec_out0_gt = self.proj_inverse(dec_output_gt)
            dec_out0_gt = dec_out0_gt.view(bn*5, 15, -1)
            gt = src_gt[:, i * 5:i * 5 + 5, :]
            gt = self.proj(gt)
            gt = gt.view(bs * 5, 15*n_person, -1)


            dec_out0_gt = dec_out0_gt.view(bn, 128, 15, 5)

            for i in range(1):
                blk = self.jtcblocks[i]
                gt = blk(gt)
            gt = gt.view(-1, 5, 128)
            for j in range(1):
                blk = self.Spatial_blocks[j]
                gt = blk(gt)
            gt = gt.view(bn, 128, 15, 5)

            concat_gtgt.append(gt)
            final_gt = torch.cat(concat_gtgt, dim=3)
         
            mpbp_seq_gt = torch.cat((mpbp_seq0_gt,final_gt),dim=3)
          
        
            src00_gt = gt.view(bn, 5, -1)

            src00_gt = self.proj_inverse1(src00_gt)
            src00_gt = src00_gt.view(bn * 5, 15, -1)
            src00_gt = self.Spatial_patch_to_embedding(src00_gt)
            src00_gt = src00_gt.view(bn ,5, 15*32)
            src0_gt=torch.cat((src0_gt,src00_gt),dim=1)
            bn,Cw,J,Tw=mpbp_seq_gt.shape
            src0_gt = self.position_encoder(src0_gt)
            mpbp_seq_gt = mpbp_seq_gt.view(bn*15, -1,128)
            mpbp_seq_gt = self.position_encoder1(mpbp_seq_gt)
            mpbp_seq_gt = self.pos_drop(mpbp_seq_gt)
            for j in range(1):
                blk = self.Spatial_blocks[j]
                mpbp_seq_gt = blk(mpbp_seq_gt)
            concat_dec_out0_gt.append(dec_out0_gt)
            mpbp_seq_gt = mpbp_seq_gt.view(bn, 128, 15, -1)
        final_dec_out0_gt = torch.cat(concat_dec_out0_gt, dim=3)
        final_dec_out0_gt = final_dec_out0_gt.view(bn, 45, -1)
        final_dec_out0_gt = self.proj_inverse1(final_dec_out0_gt)



        return final_dec_out0,final_dec_out0_gt

