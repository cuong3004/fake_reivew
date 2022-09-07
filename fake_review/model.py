from typing import Optional, Tuple, Union
from transformers import BertForSequenceClassification
import torch 
from transformers.modeling_outputs import SequenceClassifierOutput
from operator import mod
from typing import List, Optional, Tuple, Union
from numpy import dtype
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention, BertLayer, BertEncoder, BertForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
import torch.nn as nn
import numpy as np
from delight_modules.dextra_unit import DExTraUnit

class TinyBertForSequenceClassification(BertForSequenceClassification):
    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

        
from turtle import forward


class DenyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = DenyBertModel()

class DenyBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = DenyBertEncoder(config)

class DenyBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)

        assert config.encode_min_depth < config.encode_max_depth

        dextra_depths = np.linspace(start=config.encode_min_depth,
                                         stop=config.encode_max_depth,
                                         num=config.num_hidden_layers,
                                         dtype=np.int32)
        
        depth_ratio = (config.encode_max_depth * 1.0) / config.encode_min_depth

        width_multipliers = np.linspace(start=config.encode_width_mult,
                        stop=config.encode_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
                        num=config.num_hidden_layers,
                        dtype=np.float32
                        )
        
        self.layer.extend(
                [DenyBertLayer(config=config,
                                    width_multiplier=round(width_multipliers[idx], 3),
                                    dextra_depth=layer_i)
                 for idx, layer_i in enumerate(dextra_depths)
                 ]
            )
    
class DenyBertLayer(BertLayer):
    def __init__(self, config, width_multiplier, dextra_depth):
        super().__init__(config)
        self.attention = DenyBertAttention(config, width_multiplier, dextra_depth)

class DenyBertAttention(BertAttention):
    def __init__(self, config, width_multiplier, dextra_depth, position_embedding_type=None, dextra_proj=2):
        super().__init__(config, position_embedding_type)

        self.embed_dim = config.hidden_size
        assert self.embed_dim % dextra_proj == 0

        self.proj_dim = self.embed_dim // dextra_proj

        self.dextra_layer = DExTraUnit(in_features=self.embed_dim,
                                       in_proj_features=self.proj_dim,
                                       out_features=self.proj_dim,
                                       width_multiplier=width_multiplier,
                                       dextra_depth=dextra_depth,
                                       dextra_dropout=0.1,
                                       max_glt_groups=4,
                                       act_type="gelu",
                                       use_bias=True,
                                       norm_type="ln",
                                       glt_shuffle=False,
                                       is_iclr_version=False
                                       )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, output_attentions: Optional[bool] = False) -> Tuple[torch.Tensor]:
        hidden_states = self.dextra_layer(hidden_states)
        return super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
    
    