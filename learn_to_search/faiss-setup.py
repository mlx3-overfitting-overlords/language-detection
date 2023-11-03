import pandas
import torch
import faiss
from tokenizer import Tokenizer
from model import TwoTowerModel

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

def encode_passage_texts(passage_texts, tokenizer, model):
    # Encodes all passage_texts into vectors using the model and tokenizer
    passage_tensors = [torch.tensor(tokenizer.encode(passage), dtype=torch.long) for passage in passage_texts]
    with torch.no_grad():
        encoded_passage_texts = [model.encode_passage(passage_tensor.unsqueeze(0)) for passage_tensor in passage_tensors]
    return torch.stack(encoded_passage_texts).squeeze().numpy()


df = pandas.read_parquet(f'./data/data2.parquet')
data = df.query('label == 1').sample(n=10000, random_state=42).to_dict(orient='records')

passage_texts = [row['passage_text'] for row in data]
passage_urls = [row['passage_url'] for row in data]

# Encode passage_texts to create a matrix for Faiss
passage_vectors = encode_passage_texts(passage_texts, tokenizer, model)

# Normalize the vectors if they aren't already
faiss.normalize_L2(passage_vectors)

# Create a FAISS index
dimension = passage_vectors.shape[1]  # Dimensionality of your vectors
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for the similarity measure

# Add vectors to the index
index.add(passage_vectors)

faiss.write_index(index, 'passages.index')


