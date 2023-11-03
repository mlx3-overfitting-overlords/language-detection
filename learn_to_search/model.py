import torch

class CBOW(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CBOW, self).__init__()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.linear = torch.nn.Linear(embedding_dim, vocab_size)

  def forward(self, inputs):
    embeds = torch.sum(self.embeddings(inputs), dim=1)
    out = self.linear(embeds)
    log_probs = torch.nn.functional.log_softmax(out, dim=1)
    return log_probs

class RNNModel(torch.nn.Module):
    def __init__(self, embedding_matrix, hidden_dim):
        super().__init__()
        vocab_size, _ = embedding_matrix.shape
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.rnn = torch.nn.RNN(embedding_matrix.size(1), hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.rnn(embedded)
        return self.fc(output)
    

class TwoTowerModel(torch.nn.Module):
    def __init__(self, embedding_weights):
        super(TwoTowerModel, self).__init__()

        # Get the vocab size and embedding dim from the weights tensor
        vocab_size, embedding_dim = embedding_weights.shape
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        # Create an embedding layer using the pretrained weights
        self.token_embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        self.query_lstm = torch.nn.LSTM(embedding_dim, hidden_size=256, num_layers=1, batch_first=True)
        # Query Tower
        self.query_tower = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )
        
        self.passage_lstm = torch.nn.LSTM(embedding_dim, hidden_size=256, num_layers=1, batch_first=True)
        # Item/Passage Tower
        self.passage_tower = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )

        # Final prediction layer, outputs a vector of the size of the number of classes
        self.fc = torch.nn.Linear(128, 1)


    def forward(self, query, passage):
        # Embedding and processing the query sequence
        query_representation = self.encode_query(query)

        # Embedding and processing the passage sequence
        passage_representation = self.encode_passage(passage)

        # # Combining the representations
        # combined_representation = torch.cat((query_representation, passage_representation), dim=1)

        # # Final prediction layer, outputs a vector of the size of the number of classes
        # # for classification or whatever size is appropriate for your task.
        # output = self.fc(combined_representation)


        # Combining the representations with concatenation
        combined_representation = torch.cat((query_representation, passage_representation), dim=1)

        # Passing the combined vector through a dense layer to get a single score
        score = self.fc(combined_representation).squeeze()  # Assuming self.fc is a linear layer

        return score
    

    def encode_passage(self, passage):
        # Embedding and processing the passage sequence
        passage_token_embedding_sequence = self.token_embedding_layer(passage)
        _, (passage_lstm_hidden, _) = self.passage_lstm(passage_token_embedding_sequence)
        passage_representation = self.passage_tower(passage_lstm_hidden[-1])
        return passage_representation
    
    def encode_query(self, query):
        # Embedding and processing the query sequence
        query_token_embedding_sequence = self.token_embedding_layer(query)
        _, (query_lstm_hidden, _) = self.query_lstm(query_token_embedding_sequence)
        query_representation = self.query_tower(query_lstm_hidden[-1])
        return query_representation