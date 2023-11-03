import faiss
import pandas
import torch
from tokenizer import Tokenizer
from model import TwoTowerModel

# Load the FAISS index
index = faiss.read_index('passages.index')  # Make sure to provide the correct path

# Assuming you've pre-trained SentencePiece and have 'spm.model' and 'spm.vocab'
tokenizer = Tokenizer(model_path='spm.model')

# Load or initialize the TwoTowerModel with pre-trained embedding weights
vocab_size = 8000  # Vocabulary size
embedding_dim = 50  # Dimension of the embedding vector

# Normally you would load embedding weights from somewhere, but here's an example with random weights
embedding_weights = torch.randn(vocab_size, embedding_dim)
model = TwoTowerModel(embedding_weights)

# Load your model's trained weights
model.load_state_dict(torch.load('./tt_weights.pt'))
model.eval()  # Set the model to evaluation mode

def encode_passages(passages, tokenizer, model):
    # Encodes all passages into vectors using the model and tokenizer
    passage_tensors = [torch.tensor(tokenizer.encode(passage), dtype=torch.long) for passage in passages]
    with torch.no_grad():
        encoded_passages = [model.encode_passage(passage_tensor.unsqueeze(0)) for passage_tensor in passage_tensors]
    return torch.stack(encoded_passages).squeeze().numpy()


df = pandas.read_parquet(f'./data/data2.parquet')
data = df.query('label == 1').sample(n=10000, random_state=42).to_dict(orient='records')

passage_texts = [row['passage_text'] for row in data]
passage_urls = [row['passage_url'] for row in data]

def search_index(query, k=3):
    # Tokenize and encode the query
    query_tokens = torch.tensor(tokenizer.encode(query), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        query_vector = model.encode_query(query_tokens).cpu().numpy()

    # Make sure query_vector is a 2D array
    query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Search the index
    distances, indices = index.search(query_vector, k)

    # Retrieve the results
    retrieved_passages = [passage_texts[idx] for idx in indices[0]]
    retrieved_urls = [passage_urls[idx] for idx in indices[0]]

    return distances[0], retrieved_passages, retrieved_urls

# Example usage
if __name__ == "__main__":
    query = "how to use headphones with hdmi tv"
    distances, passages, urls = search_index(query)
    for distance, passage, url in zip(distances, passages, urls):
        print(f"{url} :: {passage}")
