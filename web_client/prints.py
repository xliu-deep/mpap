'''
from sentence_transformers import SentenceTransformer
psmiles_strings = ["[*]CC[*]", "[*]C=C[*]c1ccccc1"]

polyBERT = SentenceTransformer('kuelumbus/polyBERT')
embeddings = polyBERT.encode(psmiles_strings)
print(embeddings)
'''

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np 


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

'''
# Sentences we want sentence embeddings for
psmiles_strings = ["[*]CC[*]",'[*]C=C[*]c1ccccc1', "[*]CC([*])C",'[*]C=C[*]Cl','[*]OCCOC(=O)c1ccc(C([*])=O)cc1',
'[*]N1CCCCCCCC1[*]=O']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')

# Tokenize sentences
encoded_input = tokenizer(psmiles_strings, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = polyBERT(**encoded_input)

# Perform pooling. In this case, mean pooling.
fingerprints = mean_pooling(model_output, encoded_input['attention_mask'])

print(fingerprints.shape)

fingerprints=fingerprints.cpu().numpy()
fingerprints=str(fingerprints)


print("Fingerprints:")
print(fingerprints)
with open('fingerprints.txt','a')as f:
	f.write(fingerprints)
	f.close()
'''
def getprints(psmiles):
    tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
    polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')
    encoded_input = tokenizer(psmiles, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = polyBERT(**encoded_input)
    fingerprints = mean_pooling(model_output, encoded_input['attention_mask'])

    return fingerprints

