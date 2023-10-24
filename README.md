# language-detection

A language detection model.

## Getting started

### Python/conda

1. **Setup and Activate Conda Environment**

   If you haven't installed Conda yet, you can [download and install Miniconda](https://docs.conda.io/en/latest/miniconda.html) (a minimal Conda installer) or [Anaconda](https://www.anaconda.com/products/distribution) (which includes a lot more pre-installed packages).

   With Conda installed, set up a new environment and activate it:

   ```sh
   conda env create -f environment.yml
   conda activate language_detection
   ```

2. **Install Additional Dependencies (if any)**

   If you have any additional dependencies not covered by Conda, you can install them using `pip`:

   ```sh
   pip install -r requirements.txt
   ```

   > Note: It's always a good idea to keep Conda and pip dependencies separate to avoid conflicts. If you're using both, always install Conda packages first, then pip packages.

### Datasets

Download datasets by running:

```sh
./scripts/fetch_datasets.sh
```

### Jupyter Notebook

To start a Jupyter Notebook server available on any IP, run:

```sh
jupyter notebook --ip=0.0.0.0
```

### Flask app

Ensure you have Docker installed. Then, to build the container, run:

```sh
docker build -t language_detection_flask_app .
```

To start the container, run:

```sh
docker run -p 3031:3031 language_detection_flask_app
```
