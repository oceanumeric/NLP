import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import cos_sim_topn.cos_sim_topn as cs


corpus_1 = [
     'this is the first document',
     'this document is the second document',
     'and this is the third one',
     'is this the first document',
]
corpus_1 = np.array(corpus_1)

tf_idf_vect = TfidfVectorizer()  # initialize the class 
tf_idf_vect.fit(corpus_1)
skl_csr_mat = tf_idf_vect.transform(corpus_1)

# nowe we will write a python function to utilitze our c++ function
def awesome_cos_sim(A:np.ndarray, B:np.ndarray, topn:int, lower_bound=0):
    """
    Calculate the top n cosine similarities based on TF-IDF sparse matrices
    Parameters
    -----------
    A: CSR sparse matrix (M, k) made by scipy.sparse.csr_matrix 
    B: CSR sparse matrix (k, N) made by scipy.sparse.csr_matrix
    """
    A = A.tocsr()
    B = B.tocsr()
    M, ka = A.shape
    kb, N = B.shape
    assert ka == kb, \
         f"""
         The column dimension of A is {ka} does not equal to 
         the row dimension of B which is {kb};
         make sure you transposed your matrix B with feature x No. of Documents
         and matrix A and B should have the same dimension of features
         """
    idx_dtype = np.int32

    sparse_max_elements = M * topn 

    # initialize the result matrix C
    C_row_idx = np.zeros(M+1, dtype=idx_dtype)
    C_column_idx = np.zeros(sparse_max_elements, dtype=idx_dtype)
    # array type has to be double 
    C_data = np.zeros(sparse_max_elements, dtype=A.dtype)

    # check A.allclose([0]) or B.allclose([0])
    if len(A.indices) > 0 and len(A.data) > 0 and len(A.indptr) > 0 and \
        len(B.indices) > 0 and len(B.data) > 0 and len(B.indptr) > 0:
        cs.cos_sim_topn(
            M, N,
            np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            topn,
            lower_bound,
            C_row_idx, C_column_idx, C_data
        )

    # since column idx are created with grid of matrix, we keep
    # the shape as (M, N)
    return csr_matrix((C_data, C_column_idx, C_row_idx), shape=(M, N))

print(awesome_cos_sim(skl_csr_mat, skl_csr_mat.T, 3))


