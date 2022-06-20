/**
 * @file matrix_mult_dot_naive_double.cu
 *
 * CUDA code to calculate D = (A*B).*C using a naive algorithm (double precision)
 *
 */


/** Main entry point.
 * Implements naive double precision matrix multiplication and element wise multiplication
 */
__global__ void matrix_mult_dot_naive_double(
                      double * D,
                      const double * A,
                      const double * B,
		      const double * C,
                      const unsigned int A_B_sum_length,
                      const unsigned int columns_D,
                      const unsigned int rows_D) {

    // Work out which thread we are
    int row_D_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int column_D_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform matrix multiplication
    if (row_D_idx < rows_D && column_D_idx < columns_D){
        double tmp_sum = 0.0;
        for (int k_idx = 0; k_idx < A_B_sum_length; k_idx++){
            tmp_sum += A[row_D_idx * A_B_sum_length + k_idx] * B[k_idx * columns_D + column_D_idx];
        }
        // Place in the output array
        D[row_D_idx * columns_D + column_D_idx] = tmp_sum * C[row_D_idx * columns_D + column_D_idx];
    }
}

