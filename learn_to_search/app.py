from flask import Flask, request, jsonify
import torch
import faiss
from  tokenizer import Tokenizer
from model import TwoTowerModel
import pandas

app = Flask(__name__)

# These variables will act as the app state
tokenizer = None
model = None
index = None
passage_texts = []
passage_urls = []

EMBEDDING_DIM = 50

@app.before_first_request
def startup_event():
    global tokenizer, model, index, passage_texts, passage_urls
    
    # Load the tokenizer
    tokenizer = Tokenizer(model_path='spm.model')
    
    # Load or initialize the TwoTowerModel
    # Make sure you have the correct path and parameters for your model
    model = TwoTowerModel(torch.rand(len(tokenizer.vocab), EMBEDDING_DIM))
    model.load_state_dict(torch.load('./tt_weights.pt'))
    model.eval()
    
    # Load the FAISS index
    index = faiss.read_index('passages.index')
    
    # Load the data which contains the passages and URLs
    df = pandas.read_parquet('./data/data2.parquet')
    data = df.query('label == 1').to_dict(orient='records')
    passage_texts = [row['passage_text'] for row in data]
    passage_urls = [row['passage_url'] for row in data]

def search_index(query, k=10):
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

@app.route("/search", methods=['POST'])
def search():
    # Extract the query from the POST request
    query = request.json.get('query', '')
    
    # Perform the search
    distances, retrieved_passages, retrieved_urls = search_index(query)

    distances = [float(dist) for dist in distances]  # Use float(dist) to convert
    
    # Format the results into a JSON-friendly structure
    results = [
        {"distance": dist, "passage": passage, "url": url}
        for dist, passage, url in zip(distances, retrieved_passages, retrieved_urls)
    ]
    
    return jsonify(results)


@app.route("/test", methods=['GET', 'POST'])
def test():
    return jsonify({"message": "Test endpoint reached."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3031, debug=True)
