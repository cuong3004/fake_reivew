# %%
from transformers import BertModel, BertTokenizer, AdamW, AutoModel, AutoTokenizer, pipeline, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from fake_review.dataset_custom import CusttomData
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# %%
class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()

        df = pd.read_csv("fake reviews dataset.csv")
        df_train, df_test = train_test_split(
            df, random_state=42, shuffle=True, test_size=0.2
        )
        df_train, df_valid = train_test_split(
            df_train, random_state=42, shuffle=True, test_size=0.1
        )

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test




        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.acc = torchmetrics.Accuracy()

    
    def train_dataloader(self):
        dataset = CusttomData(self.df_train, self.tokenizer)
        return DataLoader(dataset, batch_size=64, num_workers=2)
    
    def val_dataloader(self):
        dataset = CusttomData(self.df_train, self.tokenizer)
        return DataLoader(dataset, batch_size=32)
    
    def test_dataloader(self):
        dataset = CusttomData(self.df_train, self.tokenizer)
        return DataLoader(dataset, batch_size=32)


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5,
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        # total_steps = len(self.train_dataloader()) * Epoch

        # # Create the learning rate scheduler.
        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                 num_warmup_steps = 0, # Default value in run_glue.py
        #                 num_training_steps = total_steps)
        # return [optimizer], [scheduler]
        return optimizer

    def share_batch(self, batch, state):
        input_ids, attention_masks, labels = batch

        out = self.model(input_ids=input_ids, 
                        attention_mask=attention_masks, 
                        labels=labels) 
        
        loss = out.loss

        # self.log('train_loss', loss)
        self.log(f"{state}_loss", loss)

        acc = self.acc(out.logits, labels)
        self.log(f'{state}_acc', acc)

        # self.log('train_acc', acc, on_step=True, on_epoch=False)
        return loss


    def training_step(self, train_batch, batch_idx):
        loss = self.share_batch(train_batch, "train")
        return loss

    def validation_step(self, val_batch, batch_idx):

        loss = self.share_batch(val_batch, "valid")

    def test_step(self, test_batch, batch_idx):

        loss = self.share_batch(test_batch, "test")
# 
# from fake_review.transformer import TinyBertForSequenceClassification, BertTokenizer

# tokenizer = BertTokenizer("../../bert-base-uncased/vocab.txt")
# %%
model_lit = LitClassification()
# %%
trainer = pl.Trainer(gpus=1, max_epochs=2)
trainer.fit(model_lit)