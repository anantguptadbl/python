###########################################
# USE TRANSFORMERS TO CREATE TRIVIA
###########################################

# MODEL IMPORT
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

#Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


## EXAMPLE 1
allLines=[]
#'https://en.wikipedia.org/wiki/Fighter_Aircraft'
with open('Fighter_Aircraft.txt') as f:
    for line in f:
        for singleSentence in line.split('.'):
            allLines.append(singleSentence)
allLines = '.'.join(allLines)

question = "What were Royal Air Force aircraft referred to"
paragraph = allLines[0:500]
encoding = tokenizer.encode_plus(text=question,text_pair=paragraph, add_special=True)

inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer = ' '.join(tokens[start_index:end_index+1])
print(answer)

## EXAMPLE 2
question = '''Why was the student group called "the Methodists?"'''

paragraph = ''' The movement which would become The United Methodist Church began in the mid-18th century within the Church of England.
            A small group of students, including John Wesley, Charles Wesley and George Whitefield, met on the Oxford University campus.
            They focused on Bible study, methodical study of scripture and living a holy life.
            Other students mocked them, saying they were the "Holy Club" and "the Methodists", being methodical and exceptionally detailed in their Bible study, opinions and disciplined lifestyle.
            Eventually, the so-called Methodists started individual societies or classes for members of the Church of England who wanted to live a more religious life. '''
            
encoding = tokenizer.encode_plus(text=question,text_pair=paragraph, add_special=True)

inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer = ' '.join(tokens[start_index:end_index+1])
print(answer)
