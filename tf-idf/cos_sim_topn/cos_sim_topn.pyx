# @ ymwdalex
# @ oceanumeric

# distutils: language = c++

import numpy as np
cimport numpy as np 


# import cpp header file 
# (.pxy should be in the same directory with header file)
cdef extern from "sparse_matrix.h":
    void sparse_cosine_sim(
                int M_row,  
                int N_col,
                int A_row_idx[],
                int A_column_idx[],
                double A_values[], 
                int B_row_idx[],
                int B_column_idx[],
                double B_values[], 
                int n, 
                double lower_bound,
                int C_row_idx[],
                int C_column_idx[],
                double C_values[]
                )

cpdef cos_sim_topn(
        int M_row,
        int N_col,
        np.ndarray[int, ndim=1] a_row_idx,
        np.ndarray[int, ndim=1] a_column_idx,
        np.ndarray[double, ndim=1] a_values,
        np.ndarray[int, ndim=1] b_row_idx,
        np.ndarray[int, ndim=1] b_column_idx,
        np.ndarray[double, ndim=1] b_values,
        int top_n,
        double lower_bound,
        np.ndarray[int, ndim=1] c_row_idx,
        np.ndarray[int, ndim=1] c_column_idx,
        np.ndarray[double, ndim=1] c_values,
    ):
    """
    Cypthon glue function to call sparse_cosine_sim function written
    in c++ function. 
    """
    # lin the pointer
    cdef int* Arow = &a_row_idx[0]
    cdef int* Acolumn = &a_column_idx[0]
    cdef double* Avalues = &a_values[0]
    cdef int* Brow = &b_row_idx[0]
    cdef int* Bcolumn = &b_column_idx[0]
    cdef double* Bvalues = &b_values[0]
    cdef int* Crow = &c_row_idx[0]
    cdef int* Ccolumn = &c_column_idx[0]
    cdef double* Cvalues = &c_values[0]

    sparse_cosine_sim(
        M_row, N_col, 
        Arow, Acolumn, Avalues, 
        Brow, Bcolumn, Bvalues, 
        top_n, lower_bound, 
        Crow, Ccolumn, Cvalues)
    
    return 
