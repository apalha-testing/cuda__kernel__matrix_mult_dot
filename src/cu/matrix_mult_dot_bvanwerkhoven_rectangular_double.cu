/**
 * The kernel is assumed to be tuned to each device by selecting
 * the best performing combination of thread block dimensions 
 * and tiling factors in X and Y. In this implementation tiling
 * in X increases the amount of work per thread block and tiling
 * in Y increases the amount of work per thread within the block. 
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */

// #define w_A_h_B 2048
// #define h_A 1024
// #define w_B 4096
/*
 * Optimized CUDA kernel for matrix multiplication
 *
 * This kernel is optimized according to the directions given
 * in: "Better performance at lower occupancy" by V. Volkov,
 * GPU Technology Conference, GTC 2010.
 *
 * The thread block dimensions (block_size_x, block_size_y) 
 * and tiling factors (tile_size_x, tile_size_y) are to be
 * tuned towards each GPU. This kernel assumes that
 * block_size_x = block_size_y * tile_size_y.
 *
 * The kernel computes D=(A*B).*C, where A, B are rectangular matrices
 * and C has the same dimensions as (A*B)
 */
__global__ void matrix_mult_dot_bvanwerkhoven_rectangular_double(double *D, double *A, double *B, double *C,
							     const unsigned int w_A_h_B,
							     const unsigned int w_B,
							     const unsigned int h_A) {

    __shared__ double sA[block_size_y*tile_size_y][block_size_x];
    __shared__ double sB[block_size_y*tile_size_y][block_size_x * tile_size_x];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * block_size_x * tile_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y * tile_size_y + threadIdx.y;
    int k, kb;

    double sum[tile_size_y][tile_size_x];
    #pragma unroll
    for (int i = 0; i < tile_size_y; i++) {
        #pragma unroll
        for (int j = 0; j < tile_size_x; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (k = 0; k < w_A_h_B; k += block_size_x) {

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < tile_size_y; i++) {
            sA[ty + block_size_y * i][tx] = A[(y+i*block_size_y) * w_A_h_B + k + tx];

            #pragma unroll
            for (int j = 0; j < tile_size_x; j++) {
                sB[ty + block_size_y * i][tx + j * block_size_x] = B[(k + ty + block_size_y * i) * w_B + x + j * block_size_x];
            }
        }
        __syncthreads();

        //compute
        #pragma unroll
        for (kb = 0; kb < block_size_x; kb++) {

            #pragma unroll
            for (int i = 0; i < tile_size_y; i++) {
            #pragma unroll
                for (int j = 0; j < tile_size_x; j++) {
                    sum[i][j] += sA[ty + block_size_y * i][kb] * sB[kb][tx + j * block_size_x];
                }
            }

        }

    }

    //store result
    #pragma unroll
    for (int i = 0; i < tile_size_y; i++) {
        #pragma unroll
        for (int j = 0; j < tile_size_x; j++) {
            D[y * w_B + x + block_size_y * i * w_B + j * block_size_x] = sum[i][j] * C[y * w_B + x + block_size_y * i * w_B + j * block_size_x];
        }
    }

}
