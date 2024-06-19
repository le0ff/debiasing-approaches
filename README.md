# Deep Neural Network Analysis: Enhance Fairness
## Debiasing Word Embeddings
This is the repository to our ([@Mangyoo](https://github.com/Mangyoo) & [@le0ff](https://github.com/le0ff)) presentation in Deep Neural Network Analysis (summer term 2024) about Enhancing Fairness. We introduce the Debiasing of Word Embeddings proposed in the paper "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings" by Tolga Bolukbasi et al. (2016).

## Overview
In the demo we will take a look at the word embedding of the Google News dataset and the (gender) bias it contains. We will analyze the gender direction, how some profession are more associated with the stereotypical female direction and others more with the stereotypical male direction and have a look at the analogy task.
Then, we will apply the proposed algorithm to debias the word embedding (Hard de-biasing, Neutralize and Equalize) and have a look at how the fairness of the word embedding improves.

## Setup
1. Clone the repository:
    ```shell
    $ git clone https://github.com/le0ff/debiasing-approaches
    ```

2. Navigate to the project directory:
    ```shell
    $ cd debiasing-approaches
    ```

3. Set up a virtual environment with Python 3.11 and install the dependencies from the `requirements.txt` file:
    ```shell
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    ```
    for windows:
    ```shell
    $ python3 -m venv venv
    $ venv\Scripts\activate
    $ pip install -r requirements.txt
    ```
    or with conda:
    ```shell
    $ conda create --name word-embedding python=3.11
    $ conda activate word-embedding
    $ pip install -r requirements.txt
    ```

4. Go to 'demo.ipynb' and try out the analysis of (gender) bias in word embedding and how the debiasing improves the word embedding.

## References

[Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). Man is to computer programmer as woman is to homemaker? debiasing word embeddings. Advances in neural information processing systems, 29.](https://proceedings.neurips.cc/paper_files/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html)

[Debiaswe: try to make word embeddings less sexist](https://github.com/tolga-b/debiaswe)