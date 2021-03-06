# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

from re import S
import torch
from torch import nn
from collections import OrderedDict
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from einops import rearrange, repeat
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

class TAggregate_img(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate_img, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        o.transpose_(1, 0)
        return o

class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 77 +1, embed_dim))
        self.type_img_embed = nn.Parameter(torch.zeros(1, clip_length, embed_dim))
        self.type_text_embed = nn.Parameter(torch.ones(1, 77, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, t_x):
        nvids = x.shape[0]
        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = x + self.type_img_embed
        t_x = t_x +  self.type_text_embed
        co_x = torch.cat((x,t_x), dim=1)
        co_x = torch.cat((cls_tokens,co_x), dim=1) + self.pos_embed
        co_x.transpose_(1, 0)
        o = self.transformer_enc(co_x)
        return o[0]


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class VideoAttText(nn.Module):
    def __init__(self, d_model: int, n_head: int, drop_out: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            # ("drop1", nn.Dropout(drop_out)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            # ("drop1", nn.Dropout(drop_out))
        ]))

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attention(self.ln_1(x), self.ln_1(y))
        x = x + self.mlp(self.ln_2(x))
        return x

class visual_prompt(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T):
        super().__init__()
        self.sim_header = sim_head
        self.T = T
        assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls", "Transf_att"]

        if self.sim_header == "LSTM" or self.sim_header == "Transf_att" or self.sim_header == "Transf" or self.sim_header == "Transf_cls" or self.sim_header == "Conv_1D" :
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            # self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
            # self.text_position_embeddings = nn.Embedding(context_length, embed_dim)
            # self.frame_type_embeddings = nn.Embedding(context_length, embed_dim)
            # self.text_type_embeddings  = nn.Embedding(context_length, embed_dim)
            # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # with torch.no_grad():
            #     trunc_normal_(self.cls_token, std=.02)

        if self.sim_header == "Transf" or  self.sim_header == "Transf_att":
            # self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            self.transformer = TAggregate(clip_length=self.T, embed_dim=embed_dim, n_layers=2)
            self.transformer_img = TAggregate_img(clip_length=self.T, embed_dim=embed_dim, n_layers=1)
            self.itm_head = nn.Linear(embed_dim, 2) 
            print('layer=6')
            # self.norm = PreNorm(embed_dim, VideoAttText(dim=embed_dim, heads = 4, dim_head = 128, dropout = 0.5))

        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(clip_length=self.T, embed_dim=embed_dim, n_layers=6)


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, t_x, list_id, logits_per_img, logits_per_text):
        b, t, c = x.size()
        x = x.contiguous()
        t_x = t_x.contiguous()

        x_original = x #(48,8, dimention(512))
        t_x_original = t_x
        # seq_length = t #t-frames 
        # text_length = t_x_original.shape[1]
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device) #tensor([0, 1, 2, 3, 4, 5, 6, 7])
        # position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1) #(batch_size, T)
        # frame_position_embeddings = self.frame_position_embeddings(position_ids)#(48,8,512)
        # frame_type_ids = torch.full_like(position_ids,1,device=x.device)
        # frame_type_embeddings = self.frame_type_embeddings(frame_type_ids)
        # text_type_ids = torch.zeros(t_x.shape[0],t_x.shape[1],dtype=torch.long, device=t_x.device)
        # text_type_embeddings = self.text_type_embeddings(text_type_ids)
        # text_position_ids = torch.arange(text_length, dtype=torch.long, device=x.device) #tensor([0, 1, 2, 3, 4, 5, 6, 7])
        # text_position_ids = text_position_ids.unsqueeze(0).expand(x.size(0), -1) #(batch_size, T)
        # text_position_embeddings = self.text_position_embeddings(text_position_ids)#(48,8,512)
        # cls_tokens = self.cls_token.expand(b, -1, -1)
        # cls_tokens_neg = self.cls_token.expand(b*2, -1, -1)

        with torch.no_grad():
            idx = list_id.reshape(1,-1)
            mask = torch.eq(idx, idx.T)
            weights_i2t = F.softmax(logits_per_img[:,:b]+1e-4,dim=1)
            weights_t2i = F.softmax(logits_per_text[:,:b]+1e-4,dim=1)
            weights_i2t.masked_fill_(mask, -1)
            weights_t2i.masked_fill_(mask, -1)

        # x = x + frame_position_embeddings + frame_type_embeddings
        # t_x = t_x + text_type_embeddings+ text_position_embeddings

    # select a negative image for each text
        image_embeds_neg = []    
        for bs in range(b): #???64????????????
            _, neg_idx = weights_t2i[bs].topk(1, dim=-1)#torch.multinomial(weights_t2i[bs], 1).item()
            image_embeds_neg.append(x[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0).squeeze(1)  
    # select a negative text for each image
        text_embeds_neg = []
        for bs in range(b):
            _, neg_idx = weights_t2i[bs].topk(1, dim=-1)
            # neg_idx = torch.multinomial(weights_i2t[bs], 1).item()
            text_embeds_neg.append(t_x[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0).squeeze(1)  
    
        text_embeds_all = torch.cat([t_x, text_embeds_neg],dim=0)   #??????????????????????????????
        image_embeds_all = torch.cat([image_embeds_neg,x],dim=0) #??????????????????????????????

# positive_forward
        x = self.transformer_img(x) 
        # x = torch.cat((cls_tokens, x), dim=1)
        # co_x = torch.cat((x,t_x), dim=1)
        # co_x = co_x.permute(1,0,2)
        co_x = self.transformer(x, t_x)
        
    # negative_forward
        # image_embeds_all = torch.cat((cls_tokens_neg, image_embeds_all), dim=1)
        # co_neg_all = torch.cat((image_embeds_all, text_embeds_all), dim=1)
        # co_neg_all = co_neg_all.permute(1,0,2)
        image_embeds_all = self.transformer_img(image_embeds_all)
        co_neg_all = self.transformer(image_embeds_all, text_embeds_all)
        
        vl_embeddings = torch.cat([co_x, co_neg_all],dim=0)
        vl_output = self.itm_head(vl_embeddings)

        return vl_output #, img_feat#out_img_feat, out_text_feat 
