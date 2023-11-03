# learn-to-search

Search Engine based upon the MS MARCO dataset

## Getting started

Download links
https://huggingface.co/datasets/ms_marco/resolve/refs%2Fconvert%2Fparquet/v1.1/test/0000.parquet

https://huggingface.co/datasets/ms_marco/resolve/refs%2Fconvert%2Fparquet/v1.1/train/0000.parquet

https://huggingface.co/datasets/ms_marco/resolve/refs%2Fconvert%2Fparquet/v1.1/validation/0000.parquet




Download datasets by running:
```sh
./scripts/fetch_datasets.sh
```


Ensure you have Docker installed, then to build the container, run:

```bash
docker build -t flask_app app .
```

To start the container, run:

```bash
docker run -p 3031:3031 app
```


### Run the following to get your data in place

```sh
python save_corpus.py
python save_data.py
# train your models
python faiss_setup.py
```