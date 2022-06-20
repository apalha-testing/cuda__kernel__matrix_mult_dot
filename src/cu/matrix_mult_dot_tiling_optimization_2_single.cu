/**
 * @file mattrix_mult_dot_tiling_optimization_2_single.cu
 * 
 * CUDA code to calculate D = (A*B).*C (optimization 2 adding loop unroll)
 * 
 */

#include <stdio.h>

// #define TILE_SIZE 32

/** Main entry point.
 * Kernel for matrix multiplication followed by elementwise dot product adding loop unroll on top of optimization 1.
 */
__global__ void matrix_mult_dot_tiling_optimization_2_single( 
                      float * D, 
                      float * A, 
                      float * B,
		      float * C,
                      const unsigned int w_A_h_B,
                      const unsigned int w_B,
                      const unsigned int h_A) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //    printf("Hello gridDim.x = %d, gridDim.y = %d\n", gridDim.x, gridDim.y);

    // Declaration of the shared memory array to store
    // sub-array A_tile
    __shared__ float A_tile[TILE_SIZE * TILE_SIZE];

    // Tile indices for array A
    // Start indices for sub-arrays of each block, maximum index, and step
    int a_begin = w_A_h_B * TILE_SIZE * by;  // index in A of the first element of first sub-array of A processed by the block
    int a_end = a_begin + w_A_h_B - 1;  // maximimum index in A of the first element of the last sub-array of A processed by the block
    int a_step = TILE_SIZE;  // the step size when jumping from the index in A of the start of a sub-array of A and the next one processed by the block
    
    // Tile indices for array B
    // Start indices for sub-arrays of each block, maximum index, and step
    // Note: each block tiles A over the columns but B over the rows, due to the
    //       nature of the matrix matrix multiplication
    int b_begin = VECTOR_SIZE * TILE_SIZE * bx;  // index in B of the first element of first sub-array of B processed by the block
    // b_end not needed because the stop of B is the same as for A, so a_end is sufficient
    int b_step = w_B * TILE_SIZE;  // the step size when jumping from the index in B of the start of a sub-array of B and the next one processed by the block  

    int c_begin = w_B * TILE_SIZE * by + VECTOR_SIZE * TILE_SIZE * bx;
    
    // Contains the elements of output array D associated to the block, but for a column
    // This is per thread
    float D_v[TILE_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // Loop over the sub-arrays of A (over the columns) and B (over the rows)
    // required to compute the block sub-array D
    // The idea is that A is subdivided into sub-arrays A_ij of size BLOCK_SIZE x BLOCKSIZE
    // the same for B, B_ij, and C, C_ij.
    // The block C_ij will then be equal to a sum of sub-array matrix matrix multiplications:
    //     C_ij = A_i1 * B_1j + A_i2 * B_2j + ... + A_ik * B_kj + ...
    // where block bx performs computation j and block by computation i and each thread computed
    // one element product and addition
    for (int a_idx = a_begin, b_idx = b_begin; a_idx <= a_end; a_idx += a_step, b_idx += b_step){

        // Load the matrices from device memory to shared memory
        // This fills the tile sub-arrays, and each thread will
        // load one element of each array
        __syncthreads();

	float *A_p = &A[a_idx + w_A_h_B * ty + tx];
	float *A_tile_p = &A_tile[ty + TILE_SIZE * tx];
#pragma unroll
	for (int n = 0; n < 16; n += 4){
	    A_tile_p[n] = A_p[w_A_h_B *n];
	}
	
	// Synchronize before computation to make sure the whole arrays are loaded
        __syncthreads();

        // Define the start of the work for each thread
	// Each thread will compute the product of a row of A_tile (column of A)
	// and an element of B. This process is repeated for all rows of A_tile and
	// all the elements in the column of B. Each thread takes care of a columns of B,
	// therefore the end product for each column is a column of C_tile.
	A_p = &A_tile[0];
	float *B_p = &B[b_idx + TILE_SIZE * ty + tx];  // threads go over the columns of B and iterate over rows

#pragma unroll
	for (int k = 0; k < TILE_SIZE; k++){
	    float bv = B_p[0];

	    D_v[0] += A_p[0] * bv;
	    D_v[1] += A_p[1] * bv;
	    D_v[2] += A_p[2] * bv;
	    D_v[3] += A_p[3] * bv;
	    D_v[4] += A_p[4] * bv;
	    D_v[5] += A_p[5] * bv;
	    D_v[6] += A_p[6] * bv;
	    D_v[7] += A_p[7] * bv;
	    D_v[8] += A_p[8] * bv;
	    D_v[9] += A_p[9] * bv;
	    D_v[10] += A_p[10] * bv;
	    D_v[11] += A_p[11] * bv;
	    D_v[12] += A_p[12] * bv;
	    D_v[13] += A_p[13] * bv;
	    D_v[14] += A_p[14] * bv;
	    D_v[15] += A_p[15] * bv;

	    A_p += TILE_SIZE;  // go on with the next row of A_tile (next column of A)
	    B_p += w_B;  // go on to the next row of B
        }
	
        // Synchronize to be sure the preceding computation is done
        __syncthreads();

    }

    // Write the block sub-array to the device memory into the
    // result array D, each thread writes one column of the sub-array
    float *D_p = &D[c_begin];
    float *C_p = &C[c_begin];
    D_p += TILE_SIZE * ty + tx;
    C_p += TILE_SIZE * ty + tx;
    int c_step = w_B;
    // Then, each thread loops over the rows of the tile of D
#pragma unroll
    for (int m=0; m < TILE_SIZE; m++){
	D_p[0] = D_v[m] * C_p[0];
	D_p += c_step;
	C_p += c_step;
    }
}
