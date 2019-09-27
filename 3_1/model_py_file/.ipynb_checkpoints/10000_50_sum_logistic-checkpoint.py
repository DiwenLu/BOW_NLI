#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data import Dataset
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import os
######################################################
## Hyper paramter


max_vocab_size = 10000
emb_dim = 50
interaction_type= "sum"
######################################################
learning_rate = 0.01
num_epochs = 10  # number epoch to train
BATCH_SIZE = 1024
filename = "{}_{}_{}_logistic".format(max_vocab_size, emb_dim, interaction_type)
######################################################
# save index 0 for unk and 1 for pad
global PAD_IDX, UNK_IDX
UNK_IDX = 0
PAD_IDX = 1


class NewsGroupDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, hypo_list, prem_list, target_list, max_sentence_length):
        """
        @param hypo_list: list of hypo tokens
        @param prem_list: list of prem tokens
        @param target_list: list of newsgroup targets
        @param max_sentence_length: fixed length of all sentence

        """
        self.hypo_list = hypo_list
        self.prem_list = prem_list
        self.target_list = target_list
        self.max_sentence_length = max_sentence_length
        assert (len(self.hypo_list) == len(self.target_list))
        assert (len(self.prem_list) == len(self.target_list))

    def __len__(self):
        return len(self.hypo_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """

        token_hypo_idx = self.hypo_list[key][:self.max_sentence_length]
        token_prem_idx = self.prem_list[key][:self.max_sentence_length]
        label = self.target_list[key]
        return [token_hypo_idx, len(token_hypo_idx), token_prem_idx, len(token_prem_idx), label]


############################################################################
# Any change about model should be here [interaction_type]
############################################################################
class LogisticRegressionPyTorch(nn.Module):
    """
    Logistic regression classification model
    Model would change according to interaction_type
    """

    def __init__(self, vocab_size, emb_dim, n_out, interaction_type):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding.
        @param n_out: size of the class.
        """
        super(LogisticRegressionPyTorch, self).__init__()

        # 1. Embedding
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # 2. Logistic Regression
        # double embedding dimension if we concat hypo's and prem's embedding
        if interaction_type == 'concat':
            emb_dim *= 2
        self.linear = nn.Linear(emb_dim, n_out)

    def forward(self, data_hypo, length_hypo, data_prem, length_prem, interaction_type):
        """
        @param data_hypo: matrix of size (batch_size, max_sentence_length).
        @param length_hypo: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data_hypo.
        @param data_prem: matrix of size (batch_size, max_sentence_length).
        @param length_hypo: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data_prem.
        @param interaction_type: [sum. hadamard, concat]
        """
        out_hypo = self.embed(data_hypo)
        out_prem = self.embed(data_prem)

        out_hypo = torch.sum(out_hypo, dim=1)
        out_prem = torch.sum(out_prem, dim=1)

        out_hypo /= length_hypo.view(length_hypo.size()[0], 1).expand_as(out_hypo).float()
        out_prem /= length_prem.view(length_prem.size()[0], 1).expand_as(out_prem).float()

        # interaction
        # 1. sum
        # 2. Hadamard product
        # 3. concat (This will change embedding dimension, 2 times as many as before)
        if interaction_type == 'concat':
            out = torch.cat((out_hypo, out_prem), 1)
        if interaction_type == 'sum':
            out = torch.add(out_hypo, out_prem)
        if interaction_type == 'hadamard':
            out = out_hypo * out_prem

        # return logits
        out = self.linear(out.float())
        return out


def build_vocab(hypo_tokens, prem_tokens, max_vocab_size):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices

    hypo_token_counter = Counter(hypo_tokens)
    prem_token_counter = Counter(prem_tokens)

    all_tokens_counter = hypo_token_counter + prem_token_counter

    vocab, count = zip(*all_tokens_counter.most_common(max_vocab_size))

    # print(all_tokens_counter.most_common(MAX_VOCAB_SIZE))

    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2, 2 + len(vocab))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token


# convert token to id in the dataset
def token2index(tokens_data):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data


def newsgroup_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    hypo_list = []
    len_hypo_list = []
    prem_list = []
    len_prem_list = []
    label_list = []

    # print("collate batch: ", batch)
    # batch[0][0] = batch[0][0][:max_sentence_length_prem]
    for datum in batch:
        label_list.append(datum[4])
        len_hypo_list.append(datum[1])
        len_prem_list.append(datum[3])
    # padding
    for datum in batch:
        # hypo
        padded_vec = np.pad(np.array(datum[0]), pad_width=((0, max_sentence_length - datum[1])), mode="constant",
                            constant_values=0)
        hypo_list.append(padded_vec)
        # prem
        padded_vec = np.pad(np.array(datum[2]), pad_width=((0, max_sentence_length - datum[3])), mode="constant",
                            constant_values=0)
        prem_list.append(padded_vec)
    return [torch.from_numpy(np.array(hypo_list)), torch.LongTensor(len_hypo_list),
            torch.from_numpy(np.array(prem_list)), torch.LongTensor(len_prem_list), torch.LongTensor(label_list)]


# Function for testing the model
def test_model(data_loader, model, interaction_type):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against

    return:
    accuracy, loss
    """
    correct = 0
    total = 0
    model.eval()

    for i, (data_hypo, lengths_hypo, data_prem, lengths_prem, labels) in enumerate(data_loader):
        outputs = model(data_hypo, lengths_hypo, data_prem, lengths_prem, interaction_type)
        # compute loss
        loss = criterion(outputs, labels)
        # compute acc
        outputs_softmax = F.softmax(outputs, dim=1)
        predicted = outputs_softmax.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total), loss.item()


if __name__ == '__main__':
    ######################################################
    # read data
    ######################################################
    # # Tokenizing be done
    folder = os.getcwd() + '/../../all_data_pickle/'
    hypo_data_tokens_train = pkl.load(open(folder + "hypo_data_tokens_train.p", "rb"))
    prem_data_tokens_train = pkl.load(open(folder + "prem_data_tokens_train.p", "rb"))
    hypo_data_tokens_val = pkl.load(open(folder + "hypo_data_tokens_val.p", "rb"))
    prem_data_tokens_val = pkl.load(open(folder + "prem_data_tokens_val.p", "rb"))
    all_hypo_data_tokens_train = pkl.load(open(folder + "all_hypo_data_tokens_train.p", "rb"))
    all_prem_data_tokens_train = pkl.load(open(folder + "all_prem_data_tokens_train.p", "rb"))
    all_hypo_data_tokens_val = pkl.load(open(folder + "all_hypo_data_tokens_val.p", "rb"))
    all_prem_data_tokens_val = pkl.load(open(folder + "all_prem_data_tokens_val.p", "rb"))
    label_index_train = pkl.load(open(folder + "label_index_train.p", "rb"))
    label_index_val = pkl.load(open(folder + "label_index_val.p", "rb"))

    # # Vocabulary

    token2id, id2token = build_vocab(all_hypo_data_tokens_train, all_prem_data_tokens_train, max_vocab_size)
    hypo_data_indices_train = token2index(hypo_data_tokens_train)
    prem_data_indices_train = token2index(prem_data_tokens_train)
    hypo_data_indices_val = token2index(hypo_data_tokens_val)
    prem_data_indices_val = token2index(prem_data_tokens_val)

    # # PyTorch DataLoader
    max_sentence_length = 20
    # trim dataset
    train_dataset = NewsGroupDataset(hypo_data_indices_train, prem_data_indices_train, label_index_train,
                                     max_sentence_length)
    val_dataset = NewsGroupDataset(hypo_data_indices_val, prem_data_indices_val, label_index_val, max_sentence_length)
    # seperate dataset into different batch
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=newsgroup_collate_func,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             collate_fn=newsgroup_collate_func,
                                             shuffle=True)

    ################################################
    # # Model
    model = LogisticRegressionPyTorch(len(id2token), emb_dim, len(set(label_index_train)), interaction_type)
    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # for optimizer: add arg "weight_decay" for regularizer (float, optional) – weight decay (L2 penalty) (default: 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []  # record training loss after every batch
    val_losses = []  # record validation loss after every batch
    train_accs = []  # record training accuracy after every batch
    val_accs = []  # record validation accuracy after every batch

    for epoch in range(num_epochs):
        print(epoch)
        running_loss_train = 0.0
        running_loss_val = 0.0
        # running_acc_train = 0.0
        running_acc_val = 0.0
        for i, (data_hypo, lengths_hypo, data_prem, lengths_prem, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = model(data_hypo, lengths_hypo, data_prem, lengths_prem, interaction_type)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            optimizer.step()

            ### training
            # training loss
            train_loss = loss.item()
            # training acc
            #train_acc, _ = test_model(train_loader, model, interaction_type)
            ### validation would be implement in each time (batch)
            val_acc, val_loss = test_model(val_loader, model, interaction_type)
            running_loss_train += train_loss
            running_loss_val += val_loss
            # running_acc_train += train_acc
            running_acc_val += val_acc

        val_accs.append(running_acc_val / len(train_loader))
        val_losses.append(running_loss_val / len(train_loader))
        # train_accs.append(running_acc_train/len(train_loader))
        train_losses.append(running_loss_train / len(train_loader))

    result = {'train_losses': train_losses,
              'val_losses': val_losses,
              # 'train_accs': train_accs,
              'val_accs': val_accs
             }

    pkl.dump(result, open("{}.p".format(os.getcwd() +'/../results_pickle_file/'+ filename), "wb"))


