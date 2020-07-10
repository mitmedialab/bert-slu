from transformers import BertForSequenceClassification
import torch
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

# # If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


output_dir="./model_save/"
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
# Copy the model to the GPU.
model.to(device)
if torch.cuda.is_available():
    model.cuda()
model.eval()
pad_len = 35 #max for sentence length
my_life_text = 'رأسي مصدع جنني سينفجر'
my_life_text = 'رأسي  جنني سينفجر'
#my_life_text = 'أشكو كثيرًا من آلام رقبتي وأحتاج حقًا إلى أن أكون أفضل'

encoded_dict = tokenizer.encode_plus(
    my_life_text,  # Sentence to encode.
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    max_length=pad_len,  # Pad & truncate all sentences.
    pad_to_max_length=True,
    return_attention_mask=True,  # Construct attn. masks.
    return_tensors='pt',  # Return pytorch tensors.
)
b_input_ids = encoded_dict['input_ids'].to(device)
b_input_mask = encoded_dict['attention_mask'].to(device)
with torch.no_grad():
    logits = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask)

results = logits[0][0].detach().cpu().numpy()
pred_flat = np.argmax(results).flatten()[0]

Arabic_intent_id_to_label = pd.read_csv('./data/Arabic_intent_id_to_label.csv', sep='\t')
Arabic_intent_id_to_label = Arabic_intent_id_to_label.set_index(['prompt_ID'])

print('You said: ', my_life_text, ' It is: ', pred_flat, ' =', Arabic_intent_id_to_label.xs(pred_flat))
