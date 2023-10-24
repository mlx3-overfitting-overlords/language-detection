# language-detection

A language detection model.

## Getting started

Download datasets by running:
```sh
./scripts/fetch_datasets.sh
```


Ensure you have Docker installed, then to build the container, run:

```bash
docker build -t language_detection_flask_app .
```

To start the container, run:

```bash
docker run -p 3031:3031 language_detection_flask_app
```
