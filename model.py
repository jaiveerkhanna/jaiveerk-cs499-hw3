# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    # https: // pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    def __init__(self, device, vocab_size,
                 embedding_dim):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = torch.nn.LSTM(
            embedding_dim, embedding_dim, batch_first=True)

    def forward(self, x):
        # Modified code from : https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # Generate embeds
        embeds = self.embedding(x)
        lstm_out = self.lstm(embeds)
        # need to extract the correct output tensor from lstm model
        lstm_final = lstm_out[1][0]
        lstm_final = lstm_final.squeeze()  # lets get rid of the extra dimension

        return lstm_final


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device,  n_actions,
                 n_targets, embedding_dim):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.n_actions = n_actions
        self.n_targets = n_targets
        # self.vocab_size = vocab_size
        self.action_embedding = nn.Embedding(n_actions, embedding_dim)
        self.target_embedding = nn.Embedding(n_targets, int(embedding_dim/2))
        self.lstm = torch.nn.LSTM(
            embedding_dim, embedding_dim, batch_first=True)

    def forward(self, input_action, input_target, hidden_decoder):

        action_embeds = self.action_embedding(input_action)
        target_embeds = self.target_embedding(input_target)

        action_embeds = action_embeds.view(
            action_embeds.shape[0], 1, self.embedding_dim)
        hidden_decoder = hidden_decoder.view(
            hidden_decoder.shape[0], 1, self.embedding_dim)
        print(action_embeds.shape)
        exit()
        lstm_out = self.lstm(action_embeds, hidden_decoder)
        exit()
        # need to extract the correct output tensor from lstm model
        lstm_final = lstm_out[1][0]
        lstm_final = lstm_final.squeeze()  # lets get rid of the extra dimension

        return lstm_final


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, vocab_size,
                 embedding_dim, max_instruction_size, n_actions,
                 n_targets):
        super(EncoderDecoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_instruction_size = max_instruction_size
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.encoder = Encoder(device, vocab_size, embedding_dim)
        self.decoder = Decoder(device, n_actions, n_targets, embedding_dim)

        # Here is where implementation changes, need 2 output FC layers
        self.fc_action = nn.Linear(embedding_dim, n_actions)
        self.fc_target = nn.Linear(embedding_dim, n_targets)

    def forward(self, x):
        N = self.max_instruction_size
        hidden_encoder = self.encoder(x)
        hidden_decoder = hidden_encoder
        pred_sequence = []
        pred_space = []
        for idx in range(N):
            action_space = self.fc_action(hidden_decoder)
            target_space = self.fc_target(hidden_decoder)

            action_pred = action_space.argmax(-1)
            target_pred = target_space.argmax(-1)

            pred_sequence.append((action_pred, target_pred))
            pred_space.append((action_space, target_space))

            # batch_size = action_space.shape[0]
            # multi_hot_encoding = torch.zeros(
            #     batch_size, self.n_actions+self.n_targets, dtype=int)

            # for idx in range(action_pred.shape[0]):
            #     multi_hot_encoding[idx][action_pred[idx]] = 1
            #     multi_hot_encoding[idx][target_pred[idx] +
            #                             self.n_actions] = 1
            # print(multi_hot_encoding)
            # print(action_pred)
            # print(target_pred)

            hidden_decoder = self.decoder(
                action_pred, target_pred, hidden_decoder)

        return pred_space, pred_sequence
