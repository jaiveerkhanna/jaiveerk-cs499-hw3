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
        encoder_outputs = lstm_out[0]
        return lstm_final, encoder_outputs


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device,  n_actions,
                 n_targets, embedding_dim, max_instruction_size):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.n_actions = n_actions
        self.n_targets = n_targets
        # self.vocab_size = vocab_size
        self.action_embedding = nn.Embedding(n_actions, int(embedding_dim/2))
        self.target_embedding = nn.Embedding(n_targets, int(embedding_dim/2))
        self.max_instruction_size = max_instruction_size
        # Attention layers
        self.attn = nn.Linear(self.embedding_dim * 2,
                              self.max_instruction_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_action, input_target, hidden_decoder):

        action_embeds = self.action_embedding(input_action)
        target_embeds = self.target_embedding(input_target)
        embeds = torch.cat((action_embeds, target_embeds), 1)

        embeds = embeds.view(
            embeds.shape[0], 1, self.embedding_dim)

        hidden_decoder = hidden_decoder.view(
            1, embeds.shape[0], -1)

        attn_weights = F.softmax(
            self.attn(torch.cat((embeds, hidden_decoder), 1)), dim=1)
        exit()

        return attn_weights, embeds


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
        self.decoder = Decoder(device, n_actions, n_targets,
                               embedding_dim, max_instruction_size)

        # Here is where implementation changes, need 2 output FC layers
        self.fc_action = nn.Linear(embedding_dim, n_actions)
        self.fc_target = nn.Linear(embedding_dim, n_targets)

        self.attn_combine = nn.Linear(
            self.embedding_dim * 2, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim, embedding_dim, batch_first=True)

    def forward(self, x):
        N = int(self.max_instruction_size)
        hidden_encoder, encoder_outputs = self.encoder(x)
        hidden_decoder = hidden_encoder
        batch_size = x.shape[0]
        # torch.zeros(batch_size, self.max_instruction_size, 2)
        pred_sequence = []
        pred_space = []

        for idx in range(N):
            action_space = self.fc_action(hidden_decoder)
            target_space = self.fc_target(hidden_decoder)

            action_pred = action_space.argmax(-1)
            target_pred = target_space.argmax(-1)

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

            attn_weights, embeds = self.decoder(
                action_pred, target_pred, hidden_decoder)

            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                     encoder_outputs.unsqueeze(0))

            output = torch.cat((embeds, attn_applied[0]), 1)
            combined_attention = self.attn_combine(output).unsqueeze(0)

            cell_state = torch.zeros(
                1, embeds.shape[0], self.embedding_dim)

            lstm_out = self.lstm(combined_attention,
                                 (hidden_decoder, cell_state))

            hidden_decoder = lstm_out[1][0].squeeze()

            pred_sequence.append((action_pred.tolist(), target_pred.tolist()))

            pred_space.append((action_space, target_space))

        return pred_space, pred_sequence
