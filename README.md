# 20 Newsgroups t-SNE Explorer
**The purpose of this app is to visually explore how different text embeddings capture document topics.**  I'm using a subset of 5 topics from the 20 Newsgroups dataset, where each document is the body of an email.  The app is hosted at: https://dash-text-tsne.herokuapp.com/ (it may take a while to launch).

This visualisation is built on top of [plotly / dash's tsne visualisation](https://github.com/plotly/dash-tsne).  Please check there for full background!


## Getting Started
### Using the demo
To get started, choose an embedding you want to visualize. When the scatter plot appears on the graph, you can see the original email text by clicking on a data point. 


### Repo contents

First create a virtual environment with conda or venv inside a temp folder, then activate it.

```
|- assets                       # CSS assets for the app
|- data                         # High dimensional vectors before TSNE (and source text as .pkl) 
|- tsne_vectors                 # 3D vectors after TSNE, for every setting 
|- app.py                       # Dash app to run
|- config.py                    # TSNE settings for generate_tsne_vectors.py and the app to display
|- demo.py                      # Detailed functions for Dash app (app.py)
|- generate_data.ipynb          # Notebook to generate /data 
|- generate_tsne_vectors.py     # Script to generate /tsne_vectors
|- README.md
|- requirements.txt
```

### Generating data
`generate_data.ipynb` and `generate_tsne_vectors.py` are included to download the data and run tsne, to use in this app.  At the moment, it generates the following embeddings for 20 Newsgroups:
* __TFIDF:__ [Term Frequency - Inverse Document Frequency vectors for each document](http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/), with SVD used to reduce dimensions from 60K+ to 200
* __Doc2Vec:__ [Document vectors generated in similar manner to Word2Vec](https://radimrehurek.com/gensim/models/doc2vec.html)
* __BERT 250 Word Embedding Mean:__ The average of [BERT word embeddings for first 250 words of a document](https://github.com/hanxiao/bert-as-service)


## About the app
### What is t-SNE?
t-distributed stochastic neighbor embedding, created by van der Maaten and Hinton in 2008, is a visualization algorithm that reduce a high-dimensional space (e.g. an image or a word embedding) into two or three dimensions, facilitating visualization of the data distribution. 

A classical example is MNIST, a dataset of 60,000 handwritten digits, 28x28 grayscale. Upon reducing the set of images using t-SNE, you can see all the digit clustered together, with few outliers caused by poor calligraphy. [You can read a detailed explanation of the algorithm on van der Maaten's personal blog.](https://lvdmaaten.github.io/tsne/)
