import time
import datetime
import random
import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
from data import train_dataset, val_dataset, tokenizer, test_dataset, intent_dim, slots_dim, tag_values
from tqdm import tqdm
from params import params
from seqeval.metrics import f1_score

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    # print('There are %d GPU(s) available.' % torch.cuda.device_count())

    # print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    # print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
# batch_size = 32
# epochs = 1
# output_dir = './model_save/'

batch_size = params.batch_size
epochs = params.epochs
output_dir = params.output_dir

# Load a trained model and vocabulary that you have fine-tuned
model = BertForTokenClassification.from_pretrained(output_dir)
# tokenizer = BertTokenizer.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)

model.eval()

if torch.cuda.is_available():
    model.cuda()



# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
# model = BertForTokenClassification.from_pretrained(
#     "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels = intent_dim+slots_dim, # The number of output labels--2 for binary classification.
#                     # You can increase this for multi-class tasks.
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
# )

# Tell pytorch to run this model on the GPU.
if torch.cuda.is_available():
    model.cuda()
    model.cuda()

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def intent_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2)[:,0].flatten()
    labels_flat = labels[:,0].flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))




# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...


# ========================================
#               Validation
# ========================================
# After the completion of each training epoch, measure our performance on
# our validation set.

# print("")
# print("Running Validation...")

t0 = time.time()

# Put the model in evaluation mode--the dropout layers behave differently
# during evaluation.
model.eval()

results = []

# Tracking variables
total_eval_accuracy = 0
total_eval_intent_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0
val_preds, val_true = [], []

# Evaluate data for one epoch
for batch in validation_dataloader:
    # Unpack this training batch from our dataloader.
    #
    # As we unpack the batch, we'll also copy each tensor to the GPU using
    # the `to` method.
    #
    # `batch` contains three pytorch tensors:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: labels
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_segment_ids = batch[2].to(device)
    b_labels = batch[3].to(device)

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        # token_type_ids is the same as the "segment ids", which
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        (loss, logits) = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)

    # Accumulate the validation loss.
    total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    total_eval_accuracy += flat_accuracy(logits, label_ids)
    total_eval_intent_accuracy += intent_accuracy(logits, label_ids)

    val_preds.extend([list(p) for p in np.argmax(logits, axis=2)])
    val_true.extend(label_ids)

    # join bpe split tokens
    for i in range(len(b_input_ids.to('cpu').numpy())):
        tokens = tokenizer.convert_ids_to_tokens(b_input_ids.to('cpu').numpy()[i])
        new_tokens, new_labels = [], []
        pred_labels = np.argmax(logits, axis=2)
        for token, label_idx in zip(tokens, pred_labels[i]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)

        results.append([" ".join(new_tokens), new_labels[0], tag_values[label_ids[i][0]]])


total_test_accuracy = 0
total_test_intent_accuracy = 0
total_test_loss = 0
nb_test_steps = 0
test_preds, test_true = [], []

# Evaluate data for one epoch
for batch in test_dataloader:
    # Unpack this training batch from our dataloader.
    #
    # As we unpack the batch, we'll also copy each tensor to the GPU using
    # the `to` method.
    #
    # `batch` contains three pytorch tensors:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: labels
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_segment_ids = batch[2].to(device)
    b_labels = batch[3].to(device)

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        # token_type_ids is the same as the "segment ids", which
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        (loss, logits) = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)

    # Accumulate the validation loss.
    total_test_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    total_test_accuracy += flat_accuracy(logits, label_ids)
    total_test_intent_accuracy += intent_accuracy(logits, label_ids)

    test_preds.extend([list(p) for p in np.argmax(logits, axis=2)])
    test_true.extend(label_ids)

    for i in range(len(b_input_ids.to('cpu').numpy())):
        tokens = tokenizer.convert_ids_to_tokens(b_input_ids.to('cpu').numpy()[i])
        new_tokens, new_labels = [], []
        pred_labels = np.argmax(logits, axis=2)
        for token, label_idx in zip(tokens, pred_labels[i]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)

        results.append([" ".join(new_tokens), new_labels[0], tag_values[label_ids[i][0]]])


val_pred_tags = [tag_values[p_i] for p, l in zip(val_preds, val_true)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
val_valid_tags = [tag_values[l_i] for l in val_true
                                  for l_i in l if tag_values[l_i] != "PAD"]
test_pred_tags = [tag_values[p_i] for p, l in zip(test_preds, test_true)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
test_valid_tags = [tag_values[l_i] for l in test_true
                                  for l_i in l if tag_values[l_i] != "PAD"]


val_f1 = f1_score(val_pred_tags, val_valid_tags)
test_f1 = f1_score(test_pred_tags, test_valid_tags)


# Report the final accuracy for this validation run.
avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
avg_test_accuracy = total_test_accuracy / len(test_dataloader)
# print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

avg_val_intent_accuracy = total_eval_intent_accuracy / len(validation_dataloader)
avg_test_intent_accuracy = total_test_intent_accuracy / len(test_dataloader)
# print("  Intent lAccuracy: {0:.2f}".format(avg_val_intent_accuracy))

# Calculate the average loss over all of the batches.
avg_val_loss = total_eval_loss / len(validation_dataloader)
avg_test_loss = total_test_loss / len(test_dataloader)

# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)

# print("  Validation Loss: {0:.2f}".format(avg_val_loss))
# print("  Validation took: {:}".format(validation_time))

print("{}\t{}\t{}\t{}\t{}".format(params.output_dir, avg_val_intent_accuracy, val_f1, avg_test_intent_accuracy, test_f1))


print("True\tPred\tUtterance")
for result in results:
    if result[1] != result[2]:
        print("{}\t{}\t{}".format(result[2], result[1], result[0].replace("[PAD]", "")))

