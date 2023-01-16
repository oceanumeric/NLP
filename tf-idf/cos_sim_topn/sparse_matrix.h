/* 
C++ header filer
declare variable types one time and use it many times in different places
this could minimize the errors of inconsistent declaration 
@ ymwdalex (github)
@ oceanumeric (github)
*/

// This is a preprocessor technique of preventing a header file 
// from being included multiple times,
#ifndef SPRSE_CPP_FILE_H  // need the endif
#define SPRSE_CPP_FILE_H

// The extern keyword is used to declare a variable or function that is 
// defined in another file, and it can be used in both header and source files.

// A function that calculate cosine similarity for a sparse matrix A 
extern void sparse_cosine_sim(
                // number of rows of sparse matrix A, len(corpus A)
                int M_row,  
                // number of columns of sparse matrix B, len(corpus B)
                int N_col,
                // sparse matrix indexes: row_pointer_array, column_idx, values
                int A_row_idx[],
                int A_column_idx[],
                double A_values[],  // value arrays of A
                int B_row_idx[],
                int B_column_idx[],
                double B_values[],  // value arrays of B
                int n,  // top n cosine similarities 
                // lower bound of cosine similarities
                // if cos_sim < lower_bound? 0 : cos_sim 
                double lower_bound,
                // sparse matrix of cosine similarities
                int C_row_idx[],
                int C_column_idx[],
                double C_values[]  // value arrays of result matrix 
                ); 


#endif  // SPARSE_CPP_FILE_H