###################################################
######## T5 MODELCONDITIONAL GENERATION ###########
###################################################

# Imports
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import tokenization_utils
import torch
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Simple prediction
input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
print(input_ids)
labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
print(labels)

# We will now check the output from the pretrained model
pred_ids = model.generate(input_ids=input_ids)
pred = ' '.join(tokenizer.decode(pred_id) for pred_id in pred_ids)
print(pred)

# Model Training
outputs = model(input_ids=input_ids, decoder_input_ids=labels)
lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
