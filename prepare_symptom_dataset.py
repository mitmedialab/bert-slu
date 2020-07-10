import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


metadata = {"pad_len": 35, "embed_dim": 300, "intent_dim": 25, "slots_dim": 1}

intent_dim = metadata['intent_dim']
slots_dim = metadata['slots_dim']
pad_len = metadata['pad_len']+1

csv_file = './data/symptom_dataset.csv'
df = pd.read_csv(csv_file)
#short dataset from 6661 sentences to 706 by removing dublicates
df = df.drop_duplicates()
df['prompt'] = pd.Categorical(df['prompt'])
df['prompt_ID'] = df.prompt.cat.codes#+1

df.to_csv('./data/symptom_dataset_short.csv', sep='\t', encoding='utf-8', index=False)

#create intent_id_to_label.pkl
df_classes = df.drop(columns=['phrase']).drop_duplicates().set_index('prompt_ID')
intent_id_to_label = df_classes.to_dict()['prompt']
#pickle.dump(intent_id_to_label, './data/symp_full/intent_id_to_label.pkl')


#create slot_id_to_label.pkl - no slots in this dataset
slot_id_to_label = {0: 'O'}
#pickle.dump(slot_id_to_label, './data/symp_full/slot_id_to_label.pkl')

sentences = df['phrase'].to_numpy()
labels = df['prompt_ID'].to_numpy()

#fixing the lables to suite the bert model
shape = np.shape(labels)
padded_labels = []
for j in range(shape[0]):
    sent = sentences[j]
    lbl = labels[j]
    num_tokens = len(sent.strip().split(' '))
    padded_labels.append(list([lbl])+list([intent_dim+1 if i < num_tokens else 0 for i in range(pad_len-1)]))

padded_labels = np.array(padded_labels)
#labels = padded_labels
#override padding
padded_labels = np.zeros((shape[0], 1))
padded_labels[:shape[0],0] = np.array(labels)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_val_index, test_index in sss.split(sentences, labels):
    train_val_data, test_data = sentences[train_val_index], sentences[test_index]
    train_val_padded_label, test_label = padded_labels[train_val_index,:], padded_labels[test_index,:]
    train_val_label = labels[train_val_index]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, val_index in sss2.split(train_val_data, train_val_label):
    train_data, val_data = train_val_data[train_index], train_val_data[val_index]
    train_label, val_label = train_val_padded_label[train_index,:], train_val_padded_label[val_index,:]

#re-write the data.py instead
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

# Load the BERT tokenizer.
# print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


tag_values = {0: "PAD"}
for k, v in intent_id_to_label.items():
    tag_values[k+1] = v
for k, v in slot_id_to_label.items():
    tag_values[k+1+intent_dim] = v

# Tokenize all of the sentences and map the tokens to thier word IDs.
def toc_sentese(sentences, labels,pad_len):
    input_ids = []
    segment_ids = []
    attention_masks = []
    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=pad_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        segment_ids.append(encoded_dict['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)
    labels = torch.cat([torch.LongTensor([l]) for l in labels], dim=0)

    dataset = TensorDataset(input_ids, attention_masks, segment_ids, labels)
    return dataset

train_dataset = toc_sentese(train_data, train_label, pad_len)
val_dataset = toc_sentese(val_data, val_label, pad_len)
test_dataset = toc_sentese(test_data, test_label, pad_len)

print('data preperation done')







