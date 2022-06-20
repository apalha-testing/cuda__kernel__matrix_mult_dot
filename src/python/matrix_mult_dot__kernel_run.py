#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Kernel tuner testing: kernel run

Description:
    Simple test to check if the CUDA kernels run within the kernel tuner.
    Tests the different version of the matrix multiplication CUDA kernels
    that implement D = (A @ B) * C (using numpy notation).

-------------------------------------------------------------------------------
Created on Fri May 27 12:31:11 2022
@author: apalha
"""

import numpy
import kernel_tuner
from collections import OrderedDict


# %% Input parameters
n_columns_A = numpy.uint32(2048) #128)  # number of columns of A and rows of B
n_rows_A = numpy.uint32(1024)#128)  # number of rows of C (and rows of A)
n_columns_B = numpy.uint32(4096)#16384)  # number of columns of C (and columns of B)
n_iterations = numpy.uint32(100)  # number of times to perform the timing operation for averaging


# %% Initialization
# Matrix sizes
n_rows_B = n_columns_A
n_rows_D = n_rows_A
n_columns_D = n_columns_B


# %% Generate arrays
# Double precision
A_double = numpy.random.rand(n_rows_A, n_columns_A)
B_double = numpy.random.rand(n_rows_B, n_columns_B)
C_double = numpy.random.rand(n_rows_D, n_columns_D)

# Single precision
A_single = A_double.astype(numpy.float32)
B_single = B_double.astype(numpy.float32)
C_single = C_double.astype(numpy.float32)

# %% Expected result
D_double = (A_double @ B_double) * C_double
D_single = (A_single @ B_single) * C_single


# %% Kernel runs

# %% Naive matrix multiplication algorithm (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_naive_double"
kernel_source = "../cu/matrix_mult_dot_naive_double.cu" 

problem_size = (n_columns_D, n_rows_D)

D_double_kernel = numpy.zeros([n_rows_D, n_columns_D])
arguments = [D_double_kernel, A_double, B_double, C_double, n_columns_A, n_columns_D, n_rows_D]

params = dict()
params["block_size_x"] = 32
params["block_size_y"] = 32

# Run kernel
print('\n---------------------------------------------------------------------')
print('Running naive matrix multiplication algorithm (double precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_double).max()

print("   Max error in D:" + str(error_D))
print("Done")
print('---------------------------------------------------------------------\n')

# %% Naive matrix multiplication algorithm (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_naive_single"
kernel_source = "../cu/matrix_mult_dot_naive_single.cu"

problem_size = (n_columns_D, n_rows_D)

D_single_kernel = numpy.zeros([n_rows_D, n_columns_D]).astype(numpy.float32)
arguments = [D_single_kernel, A_single, B_single, C_single, n_columns_A, n_columns_D, n_rows_D]

params = dict()
params["block_size_x"] = 32
params["block_size_y"] = 32

# Run kernel
print('\n---------------------------------------------------------------------')
print('Running naive matrix multiplication algorithm (single precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_single).max()

print("    Max error in D:" + str(error_D))
print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_double"
kernel_source = "../cu/matrix_mult_dot_tiling_double.cu"

problem_size = (n_columns_D, n_rows_D)

D_double_kernel = numpy.zeros([n_rows_D, n_columns_D])
arguments = [D_double_kernel, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 8
params["block_size_y"] = 8
params["TILE_SIZE"] = params["block_size_x"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm (double precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_double).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_single"
kernel_source = "../cu/matrix_mult_dot_tiling_single.cu"

problem_size = (n_columns_D, n_rows_D)

D_single_kernel = numpy.zeros([n_rows_D, n_columns_D]).astype(numpy.float32)
arguments = [D_single_kernel, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 8
params["block_size_y"] = 8
params["TILE_SIZE"] = params["block_size_x"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm (single precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_single).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm opimization 1 (no bank conflict and computation optimization) (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_1_double"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_1_double.cu"

problem_size = (n_columns_D, n_rows_D)

D_double_kernel = numpy.zeros([n_rows_D, n_columns_D])
arguments = [D_double_kernel, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 1 (double precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - D_double).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (no bank conflict and computation optimization) (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_1_single"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_1_single.cu"

problem_size = (n_columns_D, n_rows_D)

D_single_kernel = numpy.zeros([n_rows_D, n_columns_D]).astype(numpy.float32)
arguments = [D_single_kernel, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 1 (single precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - D_single).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')



# %% Tiling matrix multiplication algorithm opimization 1 (no bank conflict and computation optimization and loop unrolling) (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_2_double"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_2_double.cu"

problem_size = (n_columns_D, n_rows_D)

D_double_kernel = numpy.zeros([n_rows_D, n_columns_D])
arguments = [D_double_kernel, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 2 (double precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - D_double).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')


# %% Tiling matrix multiplication algorithm (no bank conflict and computation optimization, and loop unrolling) (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_2_single"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_2_single.cu"

problem_size = (n_columns_D, n_rows_D)

D_single_kernel = numpy.zeros([n_rows_D, n_columns_D]).astype(numpy.float32)
arguments = [D_single_kernel, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 2 (single precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - D_single).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')


# %% Tiling matrix multiplication algorithm (from bvanwerkhoven, adapted for rectangular matrices) (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_bvanwerkhoven_rectangular_double"
kernel_source = "../cu/matrix_mult_dot_bvanwerkhoven_rectangular_double.cu"

problem_size = (n_columns_D, n_rows_D)

D_double_kernel = numpy.zeros([n_rows_D, n_columns_D])
arguments = [D_double_kernel, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 8

params["tile_size_x"] = 8
params["tile_size_y"] = 2

grid_div_x = ["block_size_x", "tile_size_x"]
grid_div_y = ["block_size_y", "tile_size_y"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm rectangle bvanwerkhoven (double precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D_max = numpy.abs(answer[0] - D_double).max()
error_D_min = numpy.abs(answer[0] - D_double).min()
error_D_sum = numpy.abs(answer[0] - D_double).sum()

print("    Max error in D:" + str(error_D_max))
print("    Min error in D:" + str(error_D_min))
print("    Sum error in D:" + str(error_D_sum))

print("Done")
print('---------------------------------------------------------------------\n')


# %% Tiling matrix multiplication algorithm (from bvanwerkhoven, adapted for rectangular matrices) (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_bvanwerkhoven_rectangular_single"
kernel_source = "../cu/matrix_mult_dot_bvanwerkhoven_rectangular_single.cu"

problem_size = (n_columns_D, n_rows_D)

D_single_kernel = numpy.zeros([n_rows_D, n_columns_D]).astype(numpy.float32)
arguments = [D_single_kernel, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 32
params["block_size_y"] = 8

params["tile_size_x"] = 8
params["tile_size_y"] = 4

grid_div_x = ["block_size_x", "tile_size_x"]
grid_div_y = ["block_size_y", "tile_size_y"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm rectangle bvanwekhoven (single precision)...')
answer = kernel_tuner.run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D_max = numpy.abs(answer[0] - D_single).max()
error_D_min = numpy.abs(answer[0] - D_single).min()
error_D_sum = numpy.abs(answer[0] - D_single).sum()

print("    Max error in D:" + str(error_D_max))
print("    Min error in D:" + str(error_D_min))
print("    Sum error in D:" + str(error_D_sum))
print("    All close (atol = 1e-8, rtol = 1e-5):" + str(numpy.allclose(answer[0], D_single)))

print("Done")
print('---------------------------------------------------------------------\n')
