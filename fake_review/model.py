from typing import Optional, Tuple, Union
from transformers import BertForSequenceClassification
import torch 
from transformers.modeling_outputs import SequenceClassifierOutput

class TinyBertForSequenceClassification(BertForSequenceClassification):
    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

        
