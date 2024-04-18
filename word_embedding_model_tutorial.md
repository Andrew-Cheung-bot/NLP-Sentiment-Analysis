<Strong style="text-align:center;">This is a guide on how to download and use the FastText.</strong>
---
1. Firstly, download and release the .vec file for German from this following link:<br>[vectors-crawl](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz)

2. Then, we can load the word embedding model with the following code:

    ```python
    from gensim.models import KeyedVectors

    model = KeyedVectors.load_word2vec_format('cc.de.300.vec', binary=False)
    ```

    Example of usage:

    ```python
    print(model["hello"])
    print(model["Präzision"])
    ```

