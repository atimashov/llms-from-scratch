# Embeddings and Word Representations

Here I explore foundational techniques for learning word embeddings: **Word2Vec** and **GloVe**. These models underpin much of modern NLP and serve as a bridge toward understanding neural language models and Transformers.

## 📚 Suggested Readings

### 1. Word2Vec
- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality (Mikolov et al., 2013)](https://arxiv.org/abs/1310.4546)

### 2. Hierarchical Softmax
- [Hierarchical Probabilistic Neural Network Language Model (Morin et al., 2005)](https://proceedings.mlr.press/r5/morin05a/morin05a.pdf) – Original idea behind hierarchical softmax.

### 3. GloVe
- [GloVe: Global Vectors for Word Representation (Pennington et al., 2014)](https://nlp.stanford.edu/pubs/glove.pdf)

---

## 🧠 Word2Vec: Core Concepts

### 1. Core Idea
Predict surrounding words given a target (Skip-gram), or predict the target given its context (CBOW). Learn dense vector representations through this objective.

### 2. Hierarchical Softmax
- Original softmax is computationally expensive with large vocabularies.
- Hierarchical softmax uses a binary tree to reduce cost from \( O(V) \) to \( O(\log V) \).

### 3. Negative Sampling
- Further optimization by simplifying the prediction task to binary classification.
- Each training step updates only a few weights, improving training speed and scalability.

---

## 🧠 GloVe:  Core Concepts

Add details after implementation.
---

## 🧪 Setup and Results
Add details after implementation.

### Dependencies
```bash
pip install torch 

### Notes
To build Cython-based counter, run
python setup.py build_ext --inplace