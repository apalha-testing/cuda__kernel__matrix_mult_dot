/**
 * @file matrix_mult_dot_tiling_single.cu
 * 
 * CUDA code to calculate D = (A*B).*C (tiling algorithm, double precision)
 * 
 */

// #define TILE_SIZE 32

/** Main entry point.
 * Computes matrix multiplication using tiling algorithm followed by elementwise dot product (double precision)
 */
__global__ void matrix_mult_dot_tiling_single( 
                      float * D, 
                      const float * A, 
                      const float * B,
		      const float * C,
                      const unsigned int w_A_h_B,
                      const unsigned int w_B,
                      const unsigned int h_A) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    // Declaration of the shared memory array to store
    // sub-arrays A_tile and B_tile of A and B, respectively
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    // Tile indices for array A
    // Start indices for sub-arrays of each block, maximum index, and step
    int a_begin = w_A_h_B * TILE_SIZE * by;  // index in A of the first element of first sub-array of A processed by the block
    int a_end =  a_begin + w_A_h_B - 1;  // maximimum index in A of the first element of the last sub-array of A processed by the block
    int a_step = TILE_SIZE;  // the step size when jumping from the index in A of the start of a sub-array of A and the next one processed by the block
    
    // Tile indices for array B
    // Start indices for sub-arrays of each block, maximum index, and step
    // Note: each block tiles A over the columns but B over the rows, due to the
    //       nature of the matrix matrix multiplication
    int b_begin = TILE_SIZE * bx;  // index in B of the first element of first sub-array of B processed by the block
    // b_end not needed because the stop of B is the same as for A, so a_end is sufficient
    int b_step = w_B * TILE_SIZE;  // the step size when jumping from the index in B of the start of a sub-array of B and the next one processed by the block  

    // Contains the element of the sub-matrix of D associated
    // to the thread computation of the product between a row of A
    // and a column of B
    // This value accumulates over the tiles covered by the block
    float D_sub = 0;

    
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
	// Fill sub-arrays A_tile and B_tile with their
        // part of the original arrays
	A_tile[ty][tx] = A[a_idx + w_A_h_B * ty + tx];
        B_tile[ty][tx] = B[b_idx + w_B * ty + tx];
	
	// Synchronize before computation to make sure the whole arrays are loaded
        __syncthreads();

        // Multiply the sub-arrays together
	// Multiply a row of A_tile with a columns of B_tile
	// Each thread computes such multiplication, going over
	// the different elements of the row of A_tile and column of B_tile
        for (int k = 0; k < TILE_SIZE; k++){
            D_sub += A_tile[ty][k] * B_tile[k][tx];
        }
	
        // Synchronize to be sure the preceding computation is done
	// before loading another pair of submatrices from A and B
        __syncthreads();

    }

    // Write the block sub-array to the device memory into the
    // result array D, each thread writes one element
    int c_idx = w_B * TILE_SIZE * by + TILE_SIZE * bx; 
    D[c_idx + w_B * ty + tx] = D_sub * C[c_idx + w_B * ty + tx];
}
