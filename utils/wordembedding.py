import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


"""
WordEmbedding

Based on "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"
by Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

class WordEmbedding:
    def __init__(self, filename):
        self.thresh = None
        self.max_words = None
        self.desc = filename
        print("*** Reading data from " + filename)
        if filename.endswith(".bin"):
            import gensim.models
            model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
            words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
            vecs = [model[w] for w in words]
        else:
            vecs = []
            words = []

            with open(filename, "r", encoding='utf8') as f:
                for line in f:
                    s = line.split()
                    v = np.array([float(x) for x in s[1:]])
                    if len(vecs) and vecs[-1].shape!=v.shape:
                        print("Got weird line", line)
                        continue
    #                 v /= np.linalg.norm(v)
                    words.append(s[0])
                    vecs.append(v)
        self.vecs = np.array(vecs, dtype='float32')
        print(self.vecs.shape)
        self.words = words
        self.reindex()
        norms = np.linalg.norm(self.vecs, axis=1)
        if max(norms)-min(norms) > 0.0001:
            self.normalize()

    def reindex(self):
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n, self.d = self.vecs.shape
        assert self.n == len(self.words) == len(self.index)
        self._neighbors = None
        print(self.n, "words of dimension", self.d, ":", ", ".join(self.words[:4] + ["..."] + self.words[-4:]))

    def normalize(self):
        self.desc += ", normalize"
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
        self.reindex()

    def v(self, word):
        return self.vecs[self.index[word]]
    
    def diff(self, word1, word2):
        v = self.vecs[self.index[word1]] - self.vecs[self.index[word2]]
        return v/np.linalg.norm(v)
    
    def n_analogies(self, v, topn=500, thresh=1, max_words=50000):
        """Metric is cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0
        """
        vecs, vocab = self.vecs[:max_words], self.words[:max_words]
        self.compute_neighbors_if_necessary(thresh, max_words)
        rows, cols, vecs = self._neighbors
        scores = vecs.dot(v/np.linalg.norm(v))
        pi = np.argsort(-abs(scores))

        ans = []
        usedL = set()
        usedR = set()
        for i in pi:
            if abs(scores[i])<0.001:
                break
            row = rows[i] if scores[i] > 0 else cols[i]
            col = cols[i] if scores[i] > 0 else rows[i]
            if row in usedL or col in usedR:
                continue
            usedL.add(row)
            usedR.add(col)
            ans.append((vocab[row], vocab[col], abs(scores[i])))
            if len(ans)==topn:
                break

        return ans
    
    def compute_neighbors_if_necessary(self, thresh, max_words):
        if self._neighbors is not None and self.thresh == thresh and self.max_words == max_words:
            return
        print("Computing neighbors")
        self.thresh = thresh
        self.max_words = max_words
        vecs = self.vecs[:max_words]
        dots = vecs.dot(vecs.T)
        dots = scipy.sparse.csr_matrix(dots * (dots >= 1-thresh/2))
        from collections import Counter
        rows, cols = dots.nonzero()
        nums = list(Counter(rows).values())
        print("Mean:", np.mean(nums)-1)
        print("Median:", np.median(nums)-1)
        rows, cols, vecs = zip(*[(i, j, vecs[i]-vecs[j]) for i, j, x in zip(rows, cols, dots.data) if i<j])
        self._neighbors = rows, cols, np.array([v/np.linalg.norm(v) for v in vecs])

    def neighbors(self, word, thresh=1):
        dots = self.vecs.dot(self.v(word))
        return [self.words[i] for i, dot in enumerate(dots) if dot >= 1-thresh/2]
    
    def specific_analogy(self, a, b, c, topn=500, thresh=1, max_words=50000):
        v = self.diff(a, b) 
        all_analogies = self.n_analogies(v, topn, thresh, max_words)
        #filter analogies that do not contain c
        return [(x, y, s) for x, y, s in all_analogies if c in [x, y]]
    
    def visualize_word_embedding(self, from_idx=400, to_idx=500):
        pca = PCA(n_components=1)
        X = self.vecs[from_idx:to_idx]
        pca = PCA(n_components=2)
        pca.fit(X)

        #principal components for plotting
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]

        projections_pc1 = X.dot(pc1)
        projections_pc2 = X.dot(pc2)
        subset_words = self.words[from_idx:to_idx]

        #plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(projections_pc1, projections_pc2, alpha=0.5, label='Subset of words')

        #annotate subset words
        for word, x, y in zip(subset_words, projections_pc1, projections_pc2):
            plt.text(x, y, word, fontsize=8)

        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.xlabel('Projection on first principal component')
        plt.ylabel('Projection on second principal component')
        plt.title('Subset of Word Embeddings on Principal Component Subspace')
        plt.legend()
        plt.show()