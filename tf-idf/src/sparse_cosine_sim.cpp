/*
sparse_cosine_sim function is to calculate the cosine similarity 
given two sparse matrices A and B (in the CSR format)
we will calculate A @ B.T by iterative rows of A 
@ ymwdalex (github)
@ oceanumeric (github)
*/

#include <vector>
#include <limits>
#include <algorithm>

#include "./sparse_matrix.h"


// create a struct (mini class) for top-n cosine similarities
struct top_n_cos
{
    int index;
    double value; 
};

// a helper function to compare the value of top_n_cos
bool top_n_cos_compare(top_n_cos ci, top_n_cos cj) {
    return (ci.value > cj.value);
}; 


// the main function
// meaning of variables are given in the sparse_matrix.h file 
void sparse_cosine_sim(
                        int M_row,  
                        int N_col,
                        int A_row_idx[],
                        int A_column_idx[],
                        double A_values[],  // value arrays of A
                        int B_row_idx[],
                        int B_column_idx[],
                        double B_values[],  // value arrays of B
                        int n,  // top n cos similarities 
                        double lower_bound,
                        int C_row_idx[],  // result matrix C row idx
                        int C_column_idx[],
                        double C_values[]  // value arrays of result matrix C 
                        )
{
    // create a vector to trace columns of B (N_col)
    // it will work like a linked list
    std::vector<int> trace_B(N_col, -1);  // initial values = -1
    // vector for the dot product of M_row_i and N_col_j
    std::vector<double> dot_prod(N_col, 0);  
    std::vector<top_n_cos> top_n_cos_values; 

    // number of non zero elements in row i
    int num_non_zeros = 0;   

    // initialize the first value of c_row_idx[]
    C_row_idx[0] = 0;

    for (int i = 0; i < M_row; i++) {
        // initialize the head of linked list - trace_B
        int head = -2;
        int length = 0; 
        // get the index for column indexes array
        // remember row pointer array stores the index of column arrays
        int cj_start = A_row_idx[i];
        int cj_end = A_row_idx[i+1];

        for (int jj = cj_start; jj < cj_end; jj++){
            
        }

    } 
}

        
