import numpy as np
from sklearn.decomposition import PCA
import json
import os
import matplotlib.pyplot as plt

"""
Utils

Based on "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"
by Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

#loads professions from json file
def load_professions():
    professions_file = os.path.join(PKG_DIR, '../data', 'professions.json')
    with open(professions_file, 'r') as f:
        professions = json.load(f)
    return professions

def debias(E, gender_specific_words, definitional, equalize):
    """
    Debias the embedding E to reduce gender bias.

    E: Word Embedding to debias
    gender_specific_words: List of gender specific words
    definitional: List of pairs of words to define the gender direction
    equalize: List of pairs of words to equalize

    Returns: None (modifies the word embedding in place)
    """

    #compute gender direction using PCA on definitional word pairs
    gender_direction = compute_PC(definitional, E).components_[0]
    specific_set = set(gender_specific_words)

    #remove gender direction from vectors of non-gender specific words
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = drop(E.vecs[i], gender_direction)
    #normalize the word vectors of embedding to unit length
    E.normalize()

    #generate all variants of equalized word pairs
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    #equalize the word pairs
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            #calculate mean vector with dropped gender direction
            y = drop((E.v(a) + E.v(b)) / 2, gender_direction)
            #compute scalar to adjust vector length
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            #ensure adjustment direction based on gender direction
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            #update word vectors
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    #normalize the word vectors again after equalization
    E.normalize()

def compute_PC(pairs, embedding, num_components = 10):
    #matrix of definitional
    matrix = []
    #centering the pairs
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b))/2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    #compute PCA on matrix
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    
    #principal components of definitional gender pairs
    #plot bar plot in range of num components and explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.bar(range(num_components), pca.explained_variance_ratio_)
    plt.title("Principal Components of definitional gender pairs")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(range(num_components))
    plt.show()

    return pca

def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)