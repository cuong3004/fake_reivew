# class MyModel(nn.Module):
#     def __init__(self, out_features):
#         super().__init__()
#         self.bertmodel = AutoModel.from_pretrained(
#                     Name_model,
#                     torchscript=True)
#         self.fc = nn.Linear(768, out_features)

#     def forward(self, input_ids, attention_masks):

#         out_bert = self.bertmodel(
#                         input_ids=input_ids, 
#                         attention_mask=attention_masks,
#                         )[1]    

#         out = self.fc(out_bert)

#         return out