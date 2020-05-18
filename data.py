import random
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
import config
import json
import pickle
from os import path
from params import params

# Load the BERT tokenizer.
# print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# train_path = "/Users/matthew/Development/6.867-project/data/records/atis-eval20/train.pkl"
# val_path = "/Users/matthew/Development/6.867-project/data/records/atis-eval20/eval.pkl"
# test_path = "/Users/matthew/Development/6.867-project/data/records/atis-eval20/test.pkl"
# meta_path = "/Users/matthew/Development/6.867-project/data/records/atis-eval20/metadata.json"
# atis = True
train_path = params.train_path
val_path = params.val_path
test_path = params.test_path
meta_path = params.meta_path
atis = params.atis




with open(meta_path, "r") as f:
    metadata = json.load(f)

intent_dim = metadata['intent_dim']
slots_dim = metadata['slots_dim']
pad_len = metadata['pad_len']+1

tag_values = {0: "PAD"}

with open(test_path[:-8]+"intent_id_to_label.pkl", "rb") as f:
    intent_id_to_label = pickle.load(f)

for k, v in intent_id_to_label.items():
    tag_values[k+1] = v

with open(test_path[:-8]+"slot_id_to_label.pkl", "rb") as f:
    intent_id_to_label = pickle.load(f)

for k, v in intent_id_to_label.items():
    tag_values[k+1+intent_dim] = v

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
train_sentences = []
train_labels = []
for example in train_data:
    if atis:
        train_sentences.append(" ".join(example['tokens_text'][1:][:-1]))
        num_tokens = len(example['tokens_text'][1:][:-1])
    else:
        train_sentences.append(" ".join(example['tokens_text']))
        num_tokens = len(example['tokens_text'])
    train_labels.append(list([example['intent'][0]+1])+list([s+intent_dim+1 if i < num_tokens else 0 for i, s in enumerate(example['slots'])]))


with open(val_path, 'rb') as f:
    val_data = pickle.load(f)
val_sentences = []
val_labels = []
for example in val_data:
    if atis:
        val_sentences.append(" ".join(example['tokens_text'][1:][:-1]))
        num_tokens = len(example['tokens_text'][1:][:-1])
    else:
        val_sentences.append(" ".join(example['tokens_text']))
        num_tokens = len(example['tokens_text'])
    val_labels.append(list([example['intent'][0]+1])+list([s+intent_dim+1 if i < num_tokens else 0 for i, s in enumerate(example['slots'])]))


with open(test_path, 'rb') as f:
    test_data = pickle.load(f)
test_sentences = []
test_labels = []
for example in test_data:
    if atis:
        test_sentences.append(" ".join(example['tokens_text'][1:][:-1]))
        num_tokens = len(example['tokens_text'][1:][:-1])
    else:
        test_sentences.append(" ".join(example['tokens_text']))
        num_tokens = len(example['tokens_text'])
    test_labels.append(list([example['intent'][0]+1])+list([s+intent_dim+1 if i < num_tokens else 0 for i, s in enumerate(example['slots'])]))


# Report the number of sentences.
# print('Number of training sentences: {}\n'.format(len(train_sentences)))
# print('Number of validation sentences: {}\n'.format(len(val_sentences)))
# print('Number of test sentences: {}\n'.format(len(test_sentences)))
#
# sentences = df.sentence.values
# labels = df.label.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
train_input_ids = []
val_input_ids = []
test_input_ids = []
train_attention_masks = []
val_attention_masks = []
test_attention_masks = []
train_segment_ids = []
val_segment_ids = []
test_segment_ids = []


# For every sentence...
for sent in train_sentences:
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
    train_input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    train_attention_masks.append(encoded_dict['attention_mask'])

    train_segment_ids.append(encoded_dict['token_type_ids'])

for sent in val_sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
        max_length=pad_len,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    val_input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    val_attention_masks.append(encoded_dict['attention_mask'])

    val_segment_ids.append(encoded_dict['token_type_ids'])


for sent in test_sentences:
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
    test_input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    test_attention_masks.append(encoded_dict['attention_mask'])

    test_segment_ids.append(encoded_dict['token_type_ids'])


# Convert the lists into tensors.
train_input_ids = torch.cat(train_input_ids, dim=0)
val_input_ids = torch.cat(val_input_ids, dim=0)
test_input_ids = torch.cat(test_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
train_segment_ids = torch.cat(train_segment_ids, dim=0)
val_segment_ids = torch.cat(val_segment_ids, dim=0)
test_segment_ids = torch.cat(test_segment_ids, dim=0)
# train_labels = torch.tensor(train_labels)
# val_labels = torch.tensor(val_labels)
# test_labels = torch.tensor(test_labels)
train_labels = torch.cat([torch.LongTensor([l]) for l in train_labels], dim=0)
val_labels = torch.cat([torch.LongTensor([l]) for l in val_labels], dim=0)
test_labels = torch.cat([torch.LongTensor([l]) for l in test_labels], dim=0)


# Combine the training inputs into a TensorDataset.
# dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_segment_ids, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_segment_ids, val_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_segment_ids, test_labels)

# Create a 90-10 train-validation split.

# # Calculate the number of samples to include in each set.
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
#
# # Divide the dataset by randomly selecting samples.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print('{:>5,} training samples'.format(len(train_dataset)))
# print('{:>5,} validation samples'.format(len(val_dataset)))
# print('{:>5,} test samples'.format(len(test_dataset)))