from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, masked_lm_labels=input_ids)

loss, prediction_scores = outputs[:2]
print("Loss is {0}  prediction_scores is {1}".format(loss,prediction_scores))
