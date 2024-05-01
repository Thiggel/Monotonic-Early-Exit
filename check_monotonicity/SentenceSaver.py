from torch import nn


class SentenceSaver(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.sentence = None

    def forward(self, x):
        self.sentence = x

        return self.embedding_layer(x)
