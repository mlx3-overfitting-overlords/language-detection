from flask import Flask
import torch

app = Flask(__name__)

@app.route('/')
def hello():
    return f'Hello from Flask with PyTorch {torch.__version__}!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3031)
