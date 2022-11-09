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

        # RNN
        self.gru = nn.gru(embedding_dim, embedding_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, vocab_size, embedding_dim, n_actions,
                 n_targets):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim)

        # Here is where implementation changes, need 2 output FC layers
        self.fc_action = nn.Linear(embedding_dim, n_actions)
        self.fc_target = nn.Linear(embedding_dim, n_targets)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # Original Implementation:
        # output = self.softmax(self.out(output[0]))

        # My Implementation:
        action = self.softmax(self.fc_action(output[0]))
        target = self.softmax(self.fc_target(output[0]))

        return action, target, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self):
        pass

    def forward(self, x):
        pass
