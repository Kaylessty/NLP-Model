from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pickle
import statistics

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tqdm
import nltk
from google.colab import files

# General util functions
def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		logging.info("Creating new directory: {}".format(directory))
		os.makedirs(directory)

def print_list(l, K=None):
	# If K is given then only print first K
	for i, e in enumerate(l):
		if i == K:
			break
		print(e)
	print()

def remove_multiple_spaces(string):
	return re.sub(r'\s+', ' ', string).strip()

def save_in_pickle(save_object, save_file):
	with open(save_file, "wb") as pickle_out:
		pickle.dump(save_object, pickle_out)

def load_from_pickle(pickle_file):
	with open(pickle_file, "rb") as pickle_in:
		return pickle.load(pickle_in)

def save_in_txt(list_of_strings, save_file):
	with open(save_file, "w") as writer:
		for line in list_of_strings:
			line = line.strip()
			writer.write(f"{line}\n")

def load_from_txt(txt_file):
	with open(txt_file, "r") as reader:
		all_lines = list()
		for line in reader:
			line = line.strip()
			all_lines.append(line)
		return all_lines

print(torch.cuda.is_available())
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print("Using device:", device)

# Loading the pre-processed conversational exchanges (source-target pairs) from pickle data files
all_conversations = load_from_pickle('processed_CMDC.pkl')
# Extract 100 conversations from the end for evaluation and keep the rest for training
eval_conversations = all_conversations[-100:]
all_conversations = all_conversations[:-100]

# Logging data stats
print(f"Number of Training Conversation Pairs = {len(all_conversations)}")
print(f"Number of Evaluation Conversation Pairs = {len(eval_conversations)}")

print_list(all_conversations, 5)


pad_word = "<pad>"
bos_word = "<s>"
eos_word = "</s>"
unk_word = "<unk>"
pad_id = 0
bos_id = 1
eos_id = 2
unk_id = 3

def normalize_sentence(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class Vocabulary:
    def __init__(self):
        self.word_to_id = {pad_word: pad_id, bos_word: bos_id, eos_word:eos_id, unk_word: unk_id}
        self.word_count = {}
        self.id_to_word = {pad_id: pad_word, bos_id: bos_word, eos_id: eos_word, unk_id: unk_word}
        self.num_words = 4

    def get_ids_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        sent_ids = [bos_id] + [self.word_to_id[word] if word in self.word_to_id \
                               else unk_id for word in sentence.split()] + \
                               [eos_id]
        return sent_ids

    def tokenized_sentence(self, sentence):
        sent_ids = self.get_ids_from_sentence(sentence)
        return [self.id_to_word[word_id] for word_id in sent_ids]

    def decode_sentence_from_ids(self, sent_ids):
        words = list()
        for i, word_id in enumerate(sent_ids):
            if word_id in [bos_id, eos_id, pad_id]:
                # Skip these words
                continue
            else:
                words.append(self.id_to_word[word_id])
        return ' '.join(words)

    def add_words_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        for word in sentence.split():
            if word not in self.word_to_id:
                # add this word to the vocabulary
                self.word_to_id[word] = self.num_words
                self.id_to_word[self.num_words] = word
                self.word_count[word] = 1
                self.num_words += 1
            else:
                # update the word count
                self.word_count[word] += 1

vocab = Vocabulary()
for src, tgt in all_conversations:
    vocab.add_words_from_sentence(src)
    vocab.add_words_from_sentence(tgt)
print(f"Total words in the vocabulary = {vocab.num_words}")

print_list(sorted(vocab.word_count.items(), key=lambda item: item[1], reverse=True), 30)

for src, tgt in all_conversations[:3]:
    sentence = tgt
    word_tokens = vocab.tokenized_sentence(sentence)
    # Automatically adds bos_id and eos_id before and after sentence ids respectively
    word_ids = vocab.get_ids_from_sentence(sentence)
    print(sentence)
    print(word_tokens)
    print(word_ids)
    print(vocab.decode_sentence_from_ids(word_ids))
    print()

word = "the"
word_id = vocab.word_to_id[word]
print(f"Word = {word}")
print(f"Word ID = {word_id}")
print(f"Word decoded from ID = {vocab.decode_sentence_from_ids([word_id])}")

class SingleTurnMovieDialog_dataset(Dataset):

    def __init__(self, conversations, vocab, device):
        self.conversations = conversations
        self.vocab = vocab
        self.device = device

        def encode(src, tgt):
            src_ids = self.vocab.get_ids_from_sentence(src)
            tgt_ids = self.vocab.get_ids_from_sentence(tgt)
            return (src_ids, tgt_ids)

        # We will pre-tokenize the conversations and save in id lists for later use
        self.tokenized_conversations = [encode(src, tgt) for src, tgt in self.conversations]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"conv_ids":self.tokenized_conversations[idx], "conv":self.conversations[idx]}

def collate_fn(data):
    src_ids = [torch.LongTensor(e["conv_ids"][0]) for e in data]
    tgt_ids = [torch.LongTensor(e["conv_ids"][1]) for e in data]
    src_str = [e["conv"][0] for e in data]
    tgt_str = [e["conv"][1] for e in data]
    data = list(zip(src_ids, tgt_ids, src_str, tgt_str))
    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_ids, tgt_ids, src_str, tgt_str = zip(*data)
    src_seqs = nn.utils.rnn.pad_sequence(src_ids,batch_first=False,padding_value = pad_id)
    tgt_seqs = nn.utils.rnn.pad_sequence(tgt_ids,batch_first=False,padding_value = pad_id)

    return {"conv_ids":(src_ids, tgt_ids), "conv":(src_str, tgt_str), "conv_tensors":(src_seqs.to(device), tgt_seqs.to(device))}

# Create the DataLoader for all_conversations
dataset = SingleTurnMovieDialog_dataset(all_conversations, vocab, device)

batch_size = 5

data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                               shuffle=True, collate_fn=collate_fn)

# Test one batch of training data
first_batch = next(iter(data_loader))
print(f"Testing first training batch of size {len(first_batch['conv'][0])}")
print(f"List of source strings:")
print_list(first_batch["conv"][0])
print(f"Tokenized source ids:")
print_list(first_batch["conv_ids"][0])
print(f"Padded source ids as tensor (shape {first_batch['conv_tensors'][0].size()}):")
print(first_batch["conv_tensors"][0])

class Seq2seqBaseline(nn.Module):
    def __init__(self, vocab, emb_dim = 300, hidden_dim = 300, num_layers = 2, dropout=0.1):
        super().__init__()
        self.num_words = num_words = vocab.num_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_words, emb_dim)
        self.encoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=False)
        self.decoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=False)
        self.out = nn.Linear(hidden_dim, num_words)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode(self, source):
        source_lengths = torch.sum(source != pad_id, axis=0).cpu()
        # padCheck = (source_lengths > 0).nonzero().squeeze(-1)
        # source = source[:, padCheck]
        # source_lengths = source_lengths[padCheck]
        #print(source.size())
        mask = (source == pad_id)
        embedded = self.embedding(source)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, source_lengths, enforce_sorted=False)
        packedOut, hidden = self.encoder(packed)
        encoderOut, trash = torch.nn.utils.rnn.pad_packed_sequence(packedOut)
        hidden = torch.tanh(self.proj(torch.cat((hidden[0:hidden.size(0):2],hidden[1:hidden.size(0):2]), dim=2)))

        return encoderOut, mask, hidden


    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        del encoder_output
        del encoder_mask
        #print(decoder_input)
        # if(decoder_input.size()[1] != last_hidden.size()[1]):
        #   decoder_input = torch.cat((decoder_input[0],torch.tensor([pad_id]).to('cuda:0')))
        #   decoder_input = decoder_input.unsqueeze(0)
        embedded = self.embedding(decoder_input)
        #print(embedded)
        # print("--------")
        # print("embedded")
        # print(embedded.size())
        # print("hidden")
        # print(last_hidden.size())
        # print("-----------------------------")
        # print(decoder_input)
        # print(last_hidden)
        # print(embedded)
        # if(embedded.size()[1] != last_hidden.size()[1]):
        #   print(decoder_input.size())
        #   print(last_hidden.size())
        #   print(embedded.size())
        #   print("Oh god it happened")
        # print(embedded.size())
        # print(last_hidden.size())
        decoderOut, decoderHide = self.decoder(embedded, last_hidden)
        decoderOut = decoderOut.squeeze(0)
        log = self.out(decoderOut)
        #print(log.size())
        return log.squeeze(0), decoderHide, None


    def compute_loss(self, source, target):
        total_loss = 0
        if(source.shape[1] != target.shape[1]):
          return torch.tensor([total_loss])
        encoderOut, encoderMask, encoderHide = self.encode(source)
        decoderHide = encoderHide
        decoderIn = target[0].unsqueeze(0)
        mask = (target != pad_id)
        for i in range(1, target.size(0)):
          log, decoderHide, trash = self.decode(decoderIn, decoderHide, encoderOut, encoderMask)
          targ = target[i]
          loss = F.cross_entropy(log, targ)
          lossMask = loss * mask[i].float()
          total_loss += lossMask.sum()

          decoderIn = target[i].unsqueeze(0)
        totalToken = torch.sum(mask).float()
        total_loss = total_loss / totalToken

        return total_loss

from tqdm.notebook import trange, tqdm
def train(model, data_loader, num_epochs, model_file):
    decoder_learning_ratio = 5.0
    learning_rate = 0.000095

    encoder_parameter_names = ['embedding', 'encoder']

    encoder_named_params = list(filter(lambda kv: any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    decoder_named_params = list(filter(lambda kv: not any(key in kv[0] for key in encoder_parameter_names), model.named_parameters()))
    encoder_params = [e[1] for e in encoder_named_params]
    decoder_params = [e[1] for e in decoder_named_params]
    optimizer = torch.optim.AdamW([{'params': encoder_params},
                {'params': decoder_params, 'lr': learning_rate * decoder_learning_ratio}], lr=learning_rate)

    clip = 50.0
    for epoch in trange(num_epochs, desc="training", unit="epoch"):
        # print(f"Total training instances = {len(train_dataset)}")
        # print(f"train_data_loader = {len(train_data_loader)} {1180 > len(train_data_loader)/20}")
        with tqdm(
                data_loader,
                desc="epoch {}".format(epoch + 1),
                unit="batch",
                total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch_data in enumerate(batch_iterator, start=1):
                source, target = batch_data["conv_tensors"]
                optimizer.zero_grad()
                loss = model.compute_loss(source, target)
                total_loss += loss.item()
                loss.backward()
                # Gradient clipping before taking the step
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                batch_iterator.set_postfix(mean_loss=total_loss / i, current_loss=loss.item())
    # Save the model after training
    torch.save(model.state_dict(), model_file)

num_epochs = 6
batch_size = 64
data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                               shuffle=True, collate_fn=collate_fn)

baseline_model = Seq2seqBaseline(vocab).to(device)
from tqdm.notebook import trange, tqdm
train(baseline_model, data_loader, num_epochs, "baseline_model.pt")
files.download('baseline_model.pt')

baseline_model = Seq2seqBaseline(vocab).to(device)
baseline_model.load_state_dict(torch.load("baseline_model.pt", map_location=device))


def predict_greedy(model, sentence, max_length=100):

    model.eval()
    sentence = normalize_sentence(sentence)
    id = vocab.get_ids_from_sentence(sentence)
    # print(id)
    idTens = torch.tensor(id, dtype=torch.long, device = 'cuda:0')
    # print(idTens)
    # embed = model.embedding(idTens)
    # print(embed)
    idTens = idTens.unsqueeze(1)
    #print(idTens)
    #print(idTens.size())
    encOut, encMask, encHidden = model.encode(idTens)
    # print(encOut)
    # print(encMask)
    # print(encHidden)
    decHidden = encHidden
    # log, decoder_hidden, trash = self.decode(decoder_input, decoder_hidden, encoder_output, encoder_mask)
    decInput = torch.tensor([[1]],dtype=torch.long,device='cuda:0')
    tokens = [decInput.item()]
    for i in range(max_length):
      if len(decInput.size()) >= 3:
        decInput = decInput.squeeze(0)
      #   print("it has happened")
      # else:
      #   print("it has not happened")
      # print(decInput.size())
      # print(decHidden.size())
      # print(encOut.size())
      # print(encMask.size())
      log, decHidden, trash = model.decode(decInput,decHidden,encOut,encMask)
      topToken = log.argmax(dim=-1)
      tokens.append(topToken.item())
      if topToken.item() == eos_id:
        break
      decInput = topToken.unsqueeze(0).unsqueeze(0)
    # print(tokens)
    toRet = vocab.decode_sentence_from_ids(tokens)
    return toRet
# print(predict_greedy(baseline_model,"do you like movies?",10))

def chat_with_model(model, mode="greedy"):
    if mode == "beam":
        predict_f = predict_beam
    else:
        predict_f = predict_greedy
    chat_log = list()
    input_sentence = ''
    while(1):
        input_sentence = input('Input > ')
        if input_sentence == 'q' or input_sentence == 'quit': break

        generation = predict_f(model, input_sentence)
        if mode == "beam":
            generation = generation[0]
        print('Greedy Response:', generation)
        print()
        chat_log.append((input_sentence, generation))
    return chat_log

baseline_chat = chat_with_model(baseline_model)

from IPython.terminal.embed import embed
class Seq2seqAttention(Seq2seqBaseline):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.encAttention = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.decAttention = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.softmax = nn.Softmax(dim = -1)
        # self.attentionDec = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.attentionEnc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        # self.CC = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        embedded = self.embedding(decoder_input)
        # print(embedded.size())
        # print(last_hidden.size())
        decoderOut, decoderHide = self.decoder(embedded, last_hidden)

        encCommon = self.encAttention(encoder_output)
        decCommon = self.decAttention(decoderOut)
        # print(encCommon.size())
        # print(decCommon.size())
        scores = torch.matmul(decCommon.transpose(0,1), encCommon.permute(1, 2, 0))
        scores = self.softmax(scores)
        scores = scores.sum(dim=2)
        scores = scores.masked_fill(encoder_mask.permute(1,0),0)
        #print(scores.size())
        context = torch.matmul(scores.unsqueeze(1), encoder_output.transpose(0,1))
        #print(context.size())
        context = self.context(context)
        #print(context.size())
        final = decoderOut + context.transpose(0,1)
        #print(context.size())
        log = self.out(final.squeeze(0))
        #print(log.size())
        # print(log.size())
        # print(decoderHide.size())
        # print(scores.size())
        return log, decoderHide, scores
        # embedded = self.embedding(decoder_input)
        # decoderOut, decoderHide = self.decoder(embedded, last_hidden)
        # decoderOut = decoderOut.squeeze(0)
        # log = self.out(decoderOut)
num_epochs = 8
batch_size = 64

data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                               shuffle=True, collate_fn=collate_fn)

attention_model = Seq2seqAttention(vocab).to(device)
train(attention_model, data_loader, num_epochs, "attention_model.pt")
files.download('attention_model.pt')
attention_model = Seq2seqAttention(vocab).to(device)
attention_model.load_state_dict(torch.load("attention_model.pt", map_location=device))

def test_conversations_with_model(model, conversational_inputs = None, include_beam = False):
    basic_conversational_inputs = [
                                    "hello.",
                                    "please share you bank account number with me",
                                    "i have never met someone more annoying that you",
                                    "i like pizza. what do you like?",
                                    "give me coffee, or i'll hate you",
                                    "i'm so bored. give some suggestions",
                                    "stop running or you'll fall hard",
                                    "what is your favorite sport?",
                                    "do you believe in a miracle?",
                                    "which sport team do you like?"
    ]
    if not conversational_inputs:
        conversational_inputs = basic_conversational_inputs
    for input in conversational_inputs:
        print(f"Input > {input}")
        generation = predict_greedy(model, input)
        print('Greedy Response:', generation)
        if include_beam:
            # Also print the beam search responses from models
            generations = predict_beam(model, input)
            print('Beam Responses:')
            print_list(generations)
        print()

baseline_chat_inputs = [inp for inp, gen in baseline_chat]
attention_chat = test_conversations_with_model(attention_model, baseline_chat_inputs)
# Evaluate diversity of the models
def evaluate_diversity(model, mode="greedy"):
    if mode == "beam":
        predict_f = predict_beam
    else:
        predict_f = predict_greedy
    generations = list()
    for src, tgt in eval_conversations:
        generation = predict_f(model, src)
        if mode == "beam":
            generation = generation[0]
        generations.append(generation)
    avg_length, distinct1, distinct2 = 0, 0, 0

    unigrams = {}
    unequeUnigrams = []
    bigrams = {}
    unequeBigrams = []
    for i in generations:
      id = vocab.get_ids_from_sentence(i)
      avg_length += len(id)
      for j in id:
        if j in unigrams:
          unigrams[j] = unigrams[j] + 1
        else:
          unigrams[j] = 1
          unequeUnigrams.append(j)
      for k in range(len(id)-1):
        bi = (id[k],id[k+1])
        if bi in bigrams:
          bigrams[bi] = bigrams[bi] + 1
        else:
          bigrams[bi] = 0
          unequeBigrams.append(bi)
    totalUnigrams = 0
    for v in unequeUnigrams:
      totalUnigrams += unigrams[v]

    totalBigrams = 0
    for t in unequeBigrams:
      totalBigrams += bigrams[t]
    distinct2 = len(unequeBigrams)/totalBigrams
    distinct1 = len(unequeUnigrams)/totalUnigrams
    avg_length = avg_length/len(generations)

    # print(bigrams)
    # print(unequeBigrams)


    return avg_length, distinct1, distinct2

print(f"Baseline Model evaluation:")
avg_length, distinct1, distinct2 = evaluate_diversity(baseline_model)
print(f"Greedy decoding:")
print(f"Avg Response Length = {avg_length}")
print(f"Distinct1 = {distinct1}")
print(f"Distinct2 = {distinct2}")
print(f"Attention Model evaluation:")
avg_length, distinct1, distinct2 = evaluate_diversity(attention_model)
print(f"Greedy decoding:")
print(f"Avg Response Length = {avg_length}")
print(f"Distinct1 = {distinct1}")
print(f"Distinct2 = {distinct2}")

import pandas as pd
import numpy as np
import sys
from functools import partial
import time

!wget https://raw.githubusercontent.com/cocoxu/CS4650_projects_spring2023/master/p3_bert_train.csv
full_df = pd.read_csv('p3_bert_train.csv', header=0)


num_tweets = len(full_df)
idxs = list(range(num_tweets))
print('Total tweets in dataset: ', num_tweets)
test_idx = idxs[:int(0.1*num_tweets)]
val_idx = idxs[int(0.1*num_tweets):int(0.2*num_tweets)]
train_idx = idxs[int(0.2*num_tweets):]

train_df = full_df.iloc[train_idx].reset_index(drop=True)
val_df = full_df.iloc[val_idx].reset_index(drop=True)
test_df = full_df.iloc[test_idx].reset_index(drop=True)

train_data = train_df[['id', 'text', 'target']]
val_data   = val_df[['id', 'text', 'target']]
test_data  = test_df[['id', 'text', 'target']]


class TweetDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


train_dataset = TweetDataset(train_data)
val_dataset   = TweetDataset(val_data)
test_dataset  = TweetDataset(test_data)



def transformer_collate_fn(batch, tokenizer):
  bert_vocab = tokenizer.get_vocab()
  bert_pad_token = bert_vocab['[PAD]']
  bert_unk_token = bert_vocab['[UNK]']
  bert_cls_token = bert_vocab['[CLS]']

  sentences, labels, masks = [], [], []
  for data in batch:
    tokenizer_output = tokenizer([data['text']])
    tokenized_sent = tokenizer_output['input_ids'][0]
    mask = tokenizer_output['attention_mask'][0]
    sentences.append(torch.tensor(tokenized_sent))
    labels.append(torch.tensor(data['target']))
    masks.append(torch.tensor(mask))
  sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
  labels = torch.stack(labels, dim=0)
  masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
  return sentences, labels, masks

def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model,
          dataloader,
          optimizer,
          device,
          clip: float,
          scheduler = None):

    model.train()

    epoch_loss = 0

    for batch in dataloader:
        sentences, labels, masks = batch[0], batch[1], batch[2]

        optimizer.zero_grad()

        output = model(sentences.to(device), masks.to(device))
        loss = F.cross_entropy(output, labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler is not None:
          scheduler.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model,
             dataloader,
             device):

    model.eval()

    epoch_loss = 0
    with torch.no_grad():
      for batch in dataloader:
          sentences, labels, masks = batch[0], batch[1], batch[2]
          output = model(sentences.to(device), masks.to(device))
          loss = F.cross_entropy(output, labels.to(device))

          epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate_acc(model,
                 dataloader,
                 device):

    model.eval()

    epoch_loss = 0
    with torch.no_grad():
      total_correct = 0
      total = 0
      for i, batch in enumerate(dataloader):

          sentences, labels, masks = batch[0], batch[1], batch[2]
          output = model(sentences.to(device), masks.to(device))
          output = F.softmax(output, dim=1)
          output_class = torch.argmax(output, dim=1)
          total_correct += torch.sum(torch.where(output_class == labels.to(device), 1, 0))
          total += sentences.size()[0]

    return total_correct / total

!pip install -q transformers
from transformers import get_linear_schedule_with_warmup
from tokenizers.processors import BertProcessing


bert_model_name = 'distilbert-base-uncased'

from transformers import AutoModel, AutoTokenizer
bert_model = AutoModel.from_pretrained(bert_model_name)
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)


bert_model


class TweetClassifier(nn.Module):
    def __init__(self,
                 bert_encoder: nn.Module,
                 enc_hid_dim=768, #default embedding size
                 outputs=2,
                 dropout=0.1):
        super().__init__()

        self.bert_encoder = bert_encoder

        self.enc_hid_dim = enc_hid_dim


        ### YOUR CODE HERE ###
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(enc_hid_dim, outputs)

    def forward(self,
                src,
                mask):
        bert_output = self.bert_encoder(src, mask)

        ### YOUR CODE HERE ###
        out = bert_output.last_hidden_state[:,0]
        # print(out)
        out = self.dropout(out)
        log = self.linear(out)
        return log

def init_weights(m: nn.Module, hidden_size=768):
    k = 1/hidden_size
    for name, param in m.named_parameters():
        if 'weight' in name:
            print(name)
            nn.init.uniform_(param.data, a=-1*k**0.5, b=k**0.5)
        else:
            print(name)
            nn.init.uniform_(param.data, 0)

def init_classification_head_weights(m: nn.Module, hidden_size=768):
    ### YOUR CODE STARTS HERE ###
    k = 1/hidden_size
    for name, param in m.named_parameters():
      if 'classification_head' in name:
        if 'weight' in name:
            print(name)
            nn.init.uniform_(param.data, a=-1*k**0.5, b=k**0.5)
        else:
            print(name)
            nn.init.uniform_(param.data, 0)


#define hyperparameters
BATCH_SIZE = 10
LR = 1e-5
WEIGHT_DECAY = 0
N_EPOCHS = 3
CLIP = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TweetClassifier(bert_model).to(device)
model.apply(init_classification_head_weights)
model.to(device)
print('Model Initialized')

#create pytorch dataloaders from train_dataset, val_dataset, and test_datset
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer), shuffle = True)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer))
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer))

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=N_EPOCHS*len(train_dataloader))

print(f'The model has {count_parameters(model):,} trainable parameters')

train_loss = evaluate(model, train_dataloader, device)
train_acc = evaluate_acc(model, train_dataloader, device)

valid_loss = evaluate(model, val_dataloader, device)
valid_acc = evaluate_acc(model, val_dataloader, device)

print(f'Initial Train Loss: {train_loss:.3f}')
print(f'Initial Train Acc: {train_acc:.3f}')
print(f'Initial Valid Loss: {valid_loss:.3f}')
print(f'Initial Valid Acc: {valid_acc:.3f}')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_dataloader, optimizer, device, CLIP, scheduler)
    end_time = time.time()
    train_acc = evaluate_acc(model, train_dataloader, device)
    valid_loss = evaluate(model, val_dataloader, device)
    valid_acc = evaluate_acc(model, val_dataloader, device)
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tTrain Acc: {train_acc:.3f}')
    print(f'\tValid Loss: {valid_loss:.3f}')
    print(f'\tValid Acc: {valid_acc:.3f}')

#run this cell and save its outputs. A 75% test accuracy is needed for full credit.
test_loss = evaluate(model, test_dataloader, device)
test_acc = evaluate_acc(model, test_dataloader, device)
print(f'Test Loss: {test_loss:.3f}')
print(f'Test Acc: {test_acc:.3f}')

def predict_beam(model, sentence, k=5, max_length=100):
    alpha = 0.7
    model.eval()
    sentence = normalize_sentence(sentence)
    id = vocab.get_ids_from_sentence(sentence)
    idTens = torch.tensor(id, dtype=torch.long, device = 'cuda:0')
    idTens = idTens.unsqueeze(1)
    encOut, encMask, encHidden = model.encode(idTens)
    decInput = torch.tensor([[1]],dtype=torch.long,device='cuda:0')
    log, decHidden, trash = model.decode(decInput,encHidden,encOut,encMask)
    probs, xi = torch.topk(log,k,dim=-1)
    beams = []
    for prob, i in zip(probs.squeeze(), xi.squeeze()):
      beam = (prob.item(), [i.item()])
      beams.append(beam)
    doneBeams = []
    for i in range(1, max_length):
      curBeams = []
      for prob, beam in beams:
        if beam[-1] == eos_id:
          doneBeams.append((prob,beam))
          if len(doneBeams) == k:
            break
        else:
          decInput = torch.tensor([[beam[-1]]], dtype=torch.long, device = 'cuda:0')
          log, decHidden, trash = model.decode(decInput, decHidden, encOut, encMask)
          probs, xi = torch.topk(log,k,dim=-1)
          for curProb, yi in zip(probs.squeeze(), xi.squeeze()):
            curProb = curProb.item()
            curBeam = beam + [yi.item()]
            curScore = (prob + curProb) / (len(curBeam) ** alpha)
            curBeams.append((curScore, curBeam))
      beams = sorted(curBeams, key=lambda x: x[0], reverse=True)[:k]
    finalBeams = []
    for t in doneBeams:
      #print(t[1])
      sent = vocab.decode_sentence_from_ids(t[1])
      finalBeams.append(sent)
    #print(finalBeams[0])
    return finalBeams[:3]
#print(predict_beam(baseline_model,"do you like movies?",3))

test_conversations_with_model(baseline_model, include_beam=False)

test_conversations_with_model(baseline_model, include_beam=True)

test_conversations_with_model(attention_model, include_beam=False)

test_conversations_with_model(attention_model, include_beam=True)

print(f"Baseline Model evaluation:")
avg_length, distinct1, distinct2 = evaluate_diversity(baseline_model)
print(f"Greedy decoding:")
print(f"Avg Response Length = {avg_length}")
print(f"Distinct1 = {distinct1}")
print(f"Distinct2 = {distinct2}")
avg_length, distinct1, distinct2 = evaluate_diversity(baseline_model, mode='beam')
print(f"Beam search decoding:")
print(f"Avg Response Length = {avg_length}")
print(f"Distinct1 = {distinct1}")
print(f"Distinct2 = {distinct2}")
print(f"Attention Model evaluation:")
avg_length, distinct1, distinct2 = evaluate_diversity(attention_model,)
print(f"Greedy decoding:")
print(f"Avg Response Length = {avg_length}")
print(f"Distinct1 = {distinct1}")
print(f"Distinct2 = {distinct2}")
avg_length, distinct1, distinct2 = evaluate_diversity(attention_model, mode='beam')
print(f"Beam decoding:")
print(f"Avg Response Length = {avg_length}")
print(f"Distinct1 = {distinct1}")
print(f"Distinct2 = {distinct2}")