/*
sparse_cosine_sim function is to calculate the cosine similarity 
given two sparse matrices A and B (in the CSR format)
we will calculate A @ B by iterative rows of A 
NOTE: assume B.shape = (k, N)
@ ymwdalex (github)
@ oceanumeric (github)
*/

#include <vector>
#include <limits>
#include <algorithm>

#include "./sparse_matrix.h"


// create a struct (mini class) for top-n cosine similarities
struct candidate
{
    int index;
    double value; 
};

// a helper function to compare the value of top_n_cos
bool top_n_cos_compare(candidate ci, candidate cj) {
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
    // create a vector to trace non zeros of dot_prod
    // it will work like a linked list
    std::vector<int> trace_nonzeros(N_col, -1);  // initial values = -1
    // vector for the dot product of M_row_i and N_col_j
    std::vector<double> dot_prod(N_col, 0);  
    std::vector<candidate> top_candidates; 

    // index counter for C 
    int cidx = 0;   

    // initialize the first value of c_row_idx[]
    C_row_idx[0] = 0;

    // calculate A[i, :] @ B 
    for (int i = 0; i < M_row; i++) {
        // initialize the head of linked list - trace_B
        int head = -2;
        int num_of_nonzeros = 0; 
        // get the index for column indexes array
        // remember row pointer array stores the index of column arrays
        int cj_start = A_row_idx[i];
        int cj_end = A_row_idx[i+1];

        for (int jj = cj_start; jj < cj_end; jj++){
            // column index of A 
            int A_j = A_column_idx[jj];
            // len(A_values) = len(A_column_idx) 
            double A_value = A_values[jj];
            // since we are doing A[i, :] @ B: (M, k) (k, N)
            // B.shape = (k, N)
            // calculate the corresponding rows of B 
            // NOTE: we do not need to calculate other rows of B 
            // because entries except for A_j are all zeros
            int k_start = B_row_idx[A_j];
            int k_end = B_row_idx[A_j+1];

            for (int kk = k_start; kk < k_end; kk++) {
                // get the column index of B
                int B_j = B_column_idx[kk];
                // length(B_column_idx) = len(B_values)
                double B_value = B_values[kk]; 
                // calculate the dot product (sum of A_value * B_value)
                dot_prod[B_j] += A_value * B_value; 

                // trace the non zeros entries of dot_prod with trace_B
                if (trace_nonzeros[B_j] == -1) {
                    // update the linked list 
                    // the head of linked list is -2
                    trace_nonzeros[B_j] = head;
                    // next value will remember where it comes from
                    // in terms of inext of trace_nonzeros 
                    head = B_j;
                    num_of_nonzeros++;
                }
            }
        }

        // visit all non zero entries of dot_prod 
        // do not use i or j index 
        for (int dd = 0; dd < num_of_nonzeros; dd++){
            // no one needs a cosine similarities < 0.5 
            // last element of non-zeros 
            if (dot_prod[head] > lower_bound) {
                // append the nonzero elements
                candidate topn; 
                topn.index = head;
                topn.value = dot_prod[head]; 
                top_candidates.push_back(topn); 
            }

            int temp = head;
            // goes up to the previous value 
            head = trace_nonzeros[head];  // iterate over columns 

            trace_nonzeros[temp] = -1; // clear arrays
            dot_prod[temp] = 0; // assign the result = 0 because < lower_bound
        }

        // Now, we will select the topn candicates
        int len = (int)top_candidates.size();

        if (len > n) {
            std::partial_sort(
                top_candidates.begin(), 
                top_candidates.begin()+n,
                top_candidates.end(),
                top_n_cos_compare); 
                // update the lenght of result vector 
                len = n; 
        } else {
            std::sort(
                top_candidates.begin(),
                top_candidates.end(),
                top_n_cos_compare
            );
        }

        // update result matrix C based on sorted top_n_cos_values

        for (int cj = 0; cj < len; cj++) {
            C_column_idx[cidx] = top_candidates[cj].index;
            C_values[cidx] = top_candidates[cj].value; 
            cidx++; 
        }

        // clear top_n_cos_values for the next row
        top_candidates.clear(); 

        // update the row index for sparse matrix C

        C_row_idx[i+1] = cidx; 

    } 
}
