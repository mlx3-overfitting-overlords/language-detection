from flask import Flask, request, jsonify
import torch
import faiss
import pandas

import language_detection.model as model
import language_detection.tokenizer as tokenizer
import learn_to_search.model as learn_to_search_model_class
import learn_to_search.tokenizer as learn_to_search_tokenizer_class

app = Flask(__name__)

# These variables will act as the app state
learn_to_search_tokenizer = None
learn_to_search_model = None
learn_to_search_index = None
learn_to_search_passage_texts = []
learn_to_search_passage_urls = []

LEARN_TO_SEARCH_EMBEDDING_DIM = 50

# These variables will act as the app state, similar to app.state in FastAPI
maybe_model = None
tknz = None
lang = None
langs = ["German", "Esperanto", "French", "Italian", "Spanish", "Turkish", "English"]

@app.before_first_request
def startup_event():
    global maybe_model, tknz, lang
    maybe_model = "Attach my model to some variables"
    tknz = (tokenizer.Tokenizer()).load_vocab("./language_detection/vocab.txt")
    lang = model.Language(torch.rand(len(tknz.vocab), 50), 7)
    lang.load_state_dict(torch.load("./language_detection/weights/lang_epoch_10.pt"))
    lang.eval()

    #################################
    ### >>>> Learn to search >>>>

    global learn_to_search_tokenizer, learn_to_search_model, learn_to_search_index, learn_to_search_passage_texts, learn_to_search_passage_urls
    
    # Load the tokenizer
    learn_to_search_tokenizer = learn_to_search_tokenizer_class.Tokenizer(model_path='./learn_to_search/spm.model')
    
    # Load or initialize the TwoTowerModel
    # Make sure you have the correct path and parameters for your model
    learn_to_search_model = learn_to_search_model_class.TwoTowerModel(torch.rand(len(learn_to_search_tokenizer.vocab), LEARN_TO_SEARCH_EMBEDDING_DIM))
    learn_to_search_model.load_state_dict(torch.load('./learn_to_search/tt_weights.pt'))
    learn_to_search_model.eval()
    
    # Load the FAISS index
    learn_to_search_index = faiss.read_index('./learn_to_search/passages.index')
    
    # Load the data which contains the passages and URLs
    df = pandas.read_parquet('./learn_to_search/data/data2.parquet')
    data = df.query('label == 1').to_dict(orient='records')
    learn_to_search_passage_texts = [row['passage_text'] for row in data]
    learn_to_search_passage_urls = [row['passage_url'] for row in data]

    ### <<<< Learn to search <<<<
    #################################

@app.route("/")
def on_root():
    return { "message": "Hello App" }

@app.route("/what_language_is_this", methods=['POST'])
def on_language_challenge():
    # The POST request body has a text field, take it and tokenize it.
    # Then feed it to the language model and return the result.
    text = request.json["text"]
    tknz_encoded = tknz.encode(text)
    tknz_tensor = torch.tensor(tknz_encoded, dtype=torch.long).unsqueeze(0)
    
    if tknz_tensor.shape[1] == 0: 
        return jsonify([
            {"class": class_name, "value": 1/len(langs)}
            for class_name in langs
        ])

    lang_output = lang(tknz_tensor)
    lang_output = torch.nn.functional.softmax(lang_output, dim=1)
    lang_output = lang_output.squeeze(0).tolist()
    result = [{"class": class_name, "value": value} for class_name, value in zip(langs, lang_output)]
    return jsonify(result)

def search_index(query, k=10):
    # Tokenize and encode the query
    query_tokens = torch.tensor(learn_to_search_tokenizer.encode(query), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        query_vector = learn_to_search_model.encode_query(query_tokens).cpu().numpy()

    # Make sure query_vector is a 2D array
    query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Search the index
    distances, indices = learn_to_search_index.search(query_vector, k)

    # Retrieve the results
    retrieved_passages = [learn_to_search_passage_texts[idx] for idx in indices[0]]
    retrieved_urls = [learn_to_search_passage_urls[idx] for idx in indices[0]]

    return distances[0], retrieved_passages, retrieved_urls

@app.route("/learn_to_search", methods=['POST'])
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



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3031, debug=True)
