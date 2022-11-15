# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch
import torch.nn.functional as F


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

    def __init__(self, device, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        # self.n_actions = n_actions
        # self.n_targets = n_targets
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim, embedding_dim, batch_first=True)

    def forward(self, hidden_encoder, hidden_decoder):

        lstm_out = self.lstm(hidden_encoder, hidden_decoder)

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
        self.decoder = Decoder(device, vocab_size, embedding_dim)

        # Here is where implementation changes, need 2 output FC layers
        self.fc_action = nn.Linear(embedding_dim, n_actions)
        self.fc_target = nn.Linear(embedding_dim, n_targets)

    def forward(self, x):
        N = self.max_instruction_size
        hidden_encoder = self.encoder(x)
        hidden_decoder = hidden_encoder
        classes = []
        for idx in range(N):
            action_pred = self.fc_action(hidden_decoder)
            target_pred = self.fc_target(hidden_decoder)

            # action_pred = action_pred.argmax(-1)
            # target_pred = target_pred.argmax(-1)

            print(target_pred)
            hidden_decoder = self.decoder(
                [action_pred, target_pred], hidden_decoder)
            classes.append((action_pred, target_pred))
        return classes
