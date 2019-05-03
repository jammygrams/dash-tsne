### What am I looking at?
**The purpose of this app is to evaluate how different embeddings fare in seperating text data according to topic.** This visualisation is built on top of [plotly / dash's tsne visualisation](https://github.com/plotly/dash-tsne).

The Scatter plot above is the result of running the t-SNE algorithm on emails from a subset of the 20 Newsgroups dataset, resulting in 3D vectors that can be visualized. For demo purposes, all the data were pre-generated using limited number of input parameters, and displayed instantly. 

### How to use the demo app
To get started, choose the embedding you want to visualize:
* __TFIDF:__ [Term Frequency - Inverse Document Frequency vectors for each document](http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/), with SVD used to reduce dimensions from 60K+ to 200
* __Doc2Vec:__ [Document vectors generated in similar manner to Word2Vec](https://radimrehurek.com/gensim/models/doc2vec.html)
* __BERT 250 Word Embedding Mean:__ The average of [BERT word embeddings for first 250 words of a document](https://github.com/hanxiao/bert-as-service)

When the scatter plot appears on the graph, you can see the text of each email by clicking on a data point.


### Choosing the right parameters
The quality of a t-SNE visualization depends heavily on the input parameters when you train the algorithm. Each parameter has a great impact on how well each group of data will be clustered. Here is what you should know for each of them:
* __Number of Iterations:__ This is how many steps you want to run the algorithm. A higher number of iterations often gives better visualizations, but more time to train.
* __Perplexity:__ This is a value that influences the number of neighbors that are taken into account during the training. According to the [original paper](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf), the value should be between 5 and 50.
* __Learning Rate:__ This value determines how much weight we should give to the updates given by the algorithm at each step. It is typically between 10 and 1000.
* __Initial PCA Dimensions:__ Because the number of dimensions of the original data might be very big, we use another [dimensionality reduction technique called PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to first reduce the dataset to a smaller space, and then apply t-SNE on that space. Initial PCA dimensions of 50 has shown good experimental results in the original paper.