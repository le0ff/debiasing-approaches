import numpy as np
from sklearn.decomposition import PCA
import json
import os


PKG_DIR = os.path.dirname(os.path.abspath(__file__))
def load_professions():
    professions_file = os.path.join(PKG_DIR, '../data', 'professions.json')
    with open(professions_file, 'r') as f:
        professions = json.load(f)

    #should we remove this?
    print('Loaded professions\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return professions


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = compute_PC(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()

def compute_PC(pairs, embedding, num_components = 10):
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b))/2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    #we could plot something for the demo?
    # bar(range(num_components), pca.explained_variance_ratio_)
    return pca

def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)