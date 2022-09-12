from main import LitClassification 
import pytorch_lightning as pl

model_lit = LitClassification.load_from_checkpoint()

trainer = pl.Trainer(gpus=1, 
                    # max_epochs=16,
                    # # limit_train_batches=0.5,
                    # default_root_dir="/content/drive/MyDrive/log_fake_review/deny_bert_other",
                    # callbacks=[checkpoint_callback]
                    )
# trainer.fit(model_lit)
trainer.test(model_lit)