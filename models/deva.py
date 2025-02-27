'''
* @name: deva.py
'''

import torch
from torch import nn
from .tpf_layer import Transformer, CrossTransformer, TPFLearningEncoder
from .bert import BertTextEncoder
from einops import repeat
import torch.nn.functional as F

class DEVA(nn.Module):
    def __init__(self, dataset, MFU_depth=3, fusion_layer_depth=2, bert_pretrained='bert-base-uncased'):
        super(DEVA, self).__init__()

        # Note that when modifying T, it needs to be modified here.
        self.h_minor = nn.Parameter(torch.ones(1, 8, 128))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)

        # mosi
        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(5, 128)
            self.proj_v0 = nn.Linear(20, 128)
            self.proj_al0 = nn.Linear(768, 128)
            self.proj_vl0 = nn.Linear(768, 128)
            # self.proj_vl0 = nn.Linear(16, 128) #onehot
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(74, 128)
            self.proj_v0 = nn.Linear(35, 128)
            self.proj_al0 = nn.Linear(768, 128)
            self.proj_vl0 = nn.Linear(768, 128)
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(33, 128)
            self.proj_v0 = nn.Linear(709, 128)
            self.proj_al0 = nn.Linear(768, 128)
            self.proj_vl0 = nn.Linear(768, 128)
        else:
            assert False, "DatasetName must be mosi, mosei or sims."

        # Note that when modifying T, the Transformer needs to modify token_len
        self.proj_l = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_a = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_al = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_vl = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)

        # Note that when modifying T, text_encoder to modify num_frames, CrossTransformer to modify source_num_frames and tgt_num_frames,
        self.text_encoder = Transformer(num_frames=50, save_hidden=True, token_len=None, dim=128, depth=MFU_depth-1, heads=8, mlp_dim=128)
        self.h_tpf_layer = TPFLearningEncoder(dim=128, depth=MFU_depth, heads=8, dim_head=16, dropout = 0.)
        self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)
        self.fusion_layer_concat = nn.Linear(128+128, 128) # The final fusion module is concat.
        self.fusion_layer_add = nn.Linear(128, 128) # Finally, the fusion module is added.

        # the post_vvtfusion layers
        self.vvtfusion_dropout = nn.Dropout(p=0.1)
        self.vvtfusion_layer = nn.Linear(128+128, 128) # 'video_out': 32 + 'video_text_out': 768, 32

        # the post_aatfusion layers
        self.aatfusion_dropout = nn.Dropout(p=0.1)
        self.aatfusion_layer = nn.Linear(128+128, 128) # 'audio_out': 16 + 768, 16

        # the post_avttfusion layers
        self.avttfusion_dropout = nn.Dropout(p=0.0)
        self.avttfusion_layer = nn.Linear(128+128+128, 128) # 768+768+768, 768


        self.cls_head = nn.Sequential(
            nn.Linear(128, 1)
        )


    def forward(self, x_visual, x_audio, x_text, audio_text, vision_text):
        b = x_visual.size(0)

        h_minor = repeat(self.h_minor, '1 n d -> b n d', b = b)

        x_text = self.bertmodel(x_text)

        x_visual = self.proj_v0(x_visual)
        x_audio = self.proj_a0(x_audio)
        x_text = self.proj_l0(x_text)
        x_audio_text = self.proj_al0(audio_text)
        x_vision_text = self.proj_vl0(vision_text)

        h_v = self.proj_v(x_visual)[:, :8]
        h_a = self.proj_a(x_audio)[:, :8]
        h_t = self.proj_l(x_text)[:, :8]
        h_at = self.proj_al(x_audio_text)[:, :8]
        h_vt = self.proj_vl(x_vision_text)[:, :8]

        h_fusion_a = h_a
        h_fusion_v = h_v
        h_fusion_t = h_t

        # Combine audio and audio_text
        h_fusion_a = torch.cat([h_fusion_a, h_at], dim=-1)
        h_fusion_a = self.aatfusion_dropout(h_fusion_a)
        h_fusion_a = F.relu(self.aatfusion_layer(h_fusion_a))



        # Combining vision and vision_text
        h_fusion_v = torch.cat([h_fusion_v, h_vt], dim=-1)
        h_fusion_v = self.vvtfusion_dropout(h_fusion_v)
        h_fusion_v = F.relu(self.vvtfusion_layer(h_fusion_v))


        # Combine text and audio_text vision_text
        h_fusion_t = torch.cat([h_vt, h_at, h_t], dim=-1)
        h_fusion_t = self.avttfusion_dropout(h_fusion_t)
        h_fusion_t = F.relu(self.avttfusion_layer(h_fusion_t))



        h_t_list = self.text_encoder(h_fusion_t) 

        h_minor = self.h_tpf_layer(h_t_list, h_fusion_a, h_fusion_v, h_minor) # ([64, 8, 128])

        feat = self.fusion_layer(h_minor, h_t_list[-1])[:, 0] # ([64, 128])

        output = self.cls_head(feat)

        return output


def build_model(opt):
    if opt.datasetName == 'sims':
        l_pretrained='bert-base-chinese'
    else:
        l_pretrained='bert-base-uncased'

    model = DEVA(dataset = opt.datasetName, fusion_layer_depth=opt.fusion_layer_depth, bert_pretrained = l_pretrained)

    return model
