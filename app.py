from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route('/')
def hello():
    return f'Hello from Flask with PyTorch {torch.__version__}!'

@app.route('/what_language_is_this', methods=['POST'])
def what_language():
    # You can now work with the data
    predictions = [
        { 'class': 'German', 'value': 0.10 },
        { 'class': 'Esperanto', 'value': 0.10 },
        { 'class': 'French', 'value': 0.10 },
        { 'class': 'Italian', 'value': 0.50 },
        { 'class': 'Spanish', 'value': 0.05 },
        { 'class': 'Turkish', 'value': 0.05 },
        { 'class': 'English', 'value': 0.10 }]

    return jsonify(predictions)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=3031, debug=True)
