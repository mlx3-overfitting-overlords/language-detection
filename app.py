from flask import Flask, request, jsonify
import torch
import tokenizer
import model

app = Flask(__name__)

# These variables will act as the app state, similar to app.state in FastAPI
maybe_model = None
tknz = None
lang = None
langs = ["German", "Espanol", "French", "Italian", "Spanish", "Turkish", "English"]

@app.before_first_request
def startup_event():
    global maybe_model, tknz, lang
    maybe_model = "Attach my model to some variables"
    tknz = (tokenizer.Tokenizer()).load_vocab("./vocab.txt")
    lang = model.Language(torch.rand(len(tknz.vocab), 50), 7)
    lang.load_state_dict(torch.load("./weights/lang_epoch_3.pt"))
    lang.eval()

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3031, debug=True)
