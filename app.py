from flask import Flask
import torch
import requests
app = Flask(__name__)

@app.route('/')
def hello():
    return f'Hello from Flask with PyTorch {torch.__version__}!'

@app.route('/example-post', methods=['POST'])
def post_example():
    request_data = request.get_json()
    name = request_data['Student Name']
    course = request_data['Course']
    python_version = request_data['Test Marks']['Mathematics']
    example = request_data['Course Interested'][0]
    return '''
    The student name is: {}
    The course applied for is: {}
    The test marks for Mathematics is: {}
    The Course student is interested in is: {}'''.format(name, course, python_version, example)



@app.route('/example-get', methods=['GET'])
def get_data():
    return requests.get('http://example.com').content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3031, debug=True)
