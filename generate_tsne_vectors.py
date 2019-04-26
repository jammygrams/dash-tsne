import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import bhtsne
import numpy as np
import pandas as pd

from config import iterations_ls, perplexity_ls, pca_dim_ls, learning_rate_ls
datasets = ["tfidf","doc2vec","bert_250_word_mean"]

def generate_embedding(dataset,
                       iterations,
                       perplexity,
                       pca_dim,
                       learning_rate,
                       verbose=1,
                       mode='two_files'):
    path = f'tsne_vectors/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}'

    def display(string):
        if verbose:
            print(string)

    if os.path.exists(path):
        if os.path.exists(path + f'/data.csv'):
            display(path + ' already exists.')
            return
    else:
        os.makedirs(path)

    if mode == 'two_files':
        data = pd.read_csv(f"data/{dataset}_input.csv")
        labels = pd.read_csv(f"data/{dataset}_labels.csv")
    elif mode == 'one_file':
        data = pd.read_csv(f'data/{dataset}.csv', index_col=0, encoding="ISO-8859-1")
        labels = data.index

    nb_col = data.shape[1]

    if pca_dim is not 'none':
        pca = PCA(n_components=min(nb_col, pca_dim))
        data = pca.fit_transform(data.values)

    tsne = TSNE(n_components=3,
                n_iter=iterations,
                learning_rate=learning_rate,
                perplexity=perplexity,
                random_state=1131)

    embedding = tsne.fit_transform(data)

    # embedding = bhtsne.tsne(data, dimensions=3, perplexity=perplexity)

    embedding_df = pd.DataFrame(embedding, columns=['x', 'y', 'z'])

    embedding_df.index = np.squeeze(labels.values)

    embedding_df.to_csv(path + f'/data.csv')

    display(f'{path} has been generated.')


for dataset in datasets:
    for iterations in iterations_ls:
        for perplexity in perplexity_ls:
            for pca_dim in pca_dim_ls:
                for learning_rate in learning_rate_ls:
                    generate_embedding(dataset,
                                       iterations,
                                       perplexity,
                                       pca_dim,
                                       learning_rate,
                                       mode='two_files')
