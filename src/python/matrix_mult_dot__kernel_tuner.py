#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Kernel tuner testing, kernel tuning
    
Description: 
    Tuning of element matrix matrix multiplication CUDA kernel..
    
-------------------------------------------------------------------------------    
Created on Fri May 27 12:31:11 2022
@author: apalha
"""

import numpy
import kernel_tuner
from collections import OrderedDict

# %% Input parameters
n_columns_A = numpy.uint32(4096)#4*128)  # number of columns of A and rows of B
n_rows_A = numpy.uint32(4096)#4*128)  # number of rows of C (and rows of A)
n_columns_B = numpy.uint32(4096)#16384)  # number of columns of C (and columns of B)
n_iterations = numpy.uint32(100)  # number of times to perform the timing operation for averaging


# %% Initialization
# Matrix sizes
n_rows_B = n_columns_A
n_rows_D = n_rows_A
n_columns_D = n_columns_B


# %% Generate arrays
A_double = numpy.random.rand(n_rows_A, n_columns_A)
B_double = numpy.random.rand(n_rows_B, n_columns_B)
C_double = numpy.random.rand(n_rows_D, n_columns_D)
D_kernel_double = numpy.zeros([n_rows_D, n_columns_D])

A_single = A_double.astype(numpy.float32)
B_single = B_double.astype(numpy.float32)
C_single = C_double.astype(numpy.float32)
D_kernel_single = D_kernel_double.astype(numpy.float32)


# %% Expected result
D_double = A_double @ B_double
D_single = A_single @ B_single


# %% Kernel tuning and timing

# %% Naive matrix multiplication algorithm (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_naive_double"
kernel_source = "../cu/matrix_mult_dot_naive_double.cu" 

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_double, A_double, B_double, C_double, n_columns_A, n_columns_D, n_rows_D]

tune_params = dict()
tune_params["block_size_x"] = [8, 16, 32]
tune_params["block_size_y"] = [32, 16, 8]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : (((2*n_columns_A + 1) * n_columns_B * n_rows_A)/1e9) / (p["time"] / 1e3)

# Run kernel
print('\n---------------------------------------------------------------------')
print('Running naive matrix multiplication algorithm (double precision)...')
results, env = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, metrics=metrics)
print("Done")
print('---------------------------------------------------------------------\n')

# %% Naive matrix multiplication algorithm (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_naive_single"
kernel_source = "../cu/matrix_mult_dot_naive_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_single, A_single, B_single, C_single, n_columns_A, n_columns_D, n_rows_D]

tune_params = dict()
tune_params["block_size_x"] = [8, 16, 32]
tune_params["block_size_y"] = [32, 16, 8]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : (2*n_columns_A * n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

# Run kernel
print('\n---------------------------------------------------------------------')
print('Running naive matrix multiplication algorithm (single precision)...')
results, env  = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, metrics=metrics)
print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_double"
kernel_source = "../cu/matrix_mult_dot_tiling_double.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_double, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [4, 8, 16, 32]
tune_params["block_size_x"] = [4, 8, 16, 32]
tune_params["block_size_y"] = [4, 8, 16, 32]
restrict = ["block_size_x==block_size_y", "TILE_SIZE==block_size_x"]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : ((2*n_columns_A + 1) * n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm (double precision)...')
results, env = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, restrictions=restrict, metrics=metrics)
print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_single"
kernel_source = "../cu/matrix_mult_dot_tiling_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_single, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [4, 8, 16, 32]
tune_params["block_size_x"] = [4, 8, 16, 32]
tune_params["block_size_y"] = [4, 8, 16, 32]
restrict = ["block_size_x==block_size_y", "TILE_SIZE==block_size_x"]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : ((2*n_columns_A + 1) * n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm (single precision)...')
results, env = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, restrictions=restrict, metrics=metrics)
print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm opimization 1 (no bank conflict and computation optimization) (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_1_double"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_1_double.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_double, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

# Here just  check the speed, the kernel cannot yet be tunable, needs changes
tune_params = dict()
tune_params["TILE_SIZE"] = [16]
tune_params["block_size_x"] = [16]
tune_params["block_size_y"] = [4]
tune_params["VECTOR_SIZE"] = [4]
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : ((2*n_columns_A + 1) * n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 1 (double precision)...')
results, env = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, grid_div_x = grid_div_x, grid_div_y = grid_div_y, verbose=True, metrics=metrics)
print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (no bank conflict and computation optimization) (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_1_single"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_1_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_single, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

# Here just  check the speed, the kernel cannot yet be tunable, needs changes
tune_params = dict()
tune_params["TILE_SIZE"] = [16]
tune_params["block_size_x"] = [16]
tune_params["block_size_y"] = [4]
tune_params["VECTOR_SIZE"] = [4]
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : ((2*n_columns_A + 1) * n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 1 (single precision)...')
results, env = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, grid_div_x = grid_div_x, grid_div_y = grid_div_y, verbose=True, metrics=metrics)
print("Done")
print('---------------------------------------------------------------------\n')



# %% Tiling matrix multiplication algorithm opimization 1 (no bank conflict and computation optimization and loop unrolling) (double precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_2_double"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_2_double.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_double, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [16]
tune_params["block_size_x"] = [16]
tune_params["block_size_y"] = [4]
tune_params["VECTOR_SIZE"] = [4]
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : ((2*n_columns_A + 1) * n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 2 (double precision)...')
results, env = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, grid_div_x = grid_div_x, grid_div_y = grid_div_y, verbose=True, metrics=metrics)
print("Done")
print('---------------------------------------------------------------------\n')


# %% Tiling matrix multiplication algorithm (no bank conflict and computation optimization, and loop unrolling) (single precision)
# Setup kernel
kernel_name = "matrix_mult_dot_tiling_optimization_2_single"
kernel_source = "../cu/matrix_mult_dot_tiling_optimization_2_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_single, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [16]
tune_params["block_size_x"] = [16]
tune_params["block_size_y"] = [4]
tune_params["VECTOR_SIZE"] = [4]
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

metrics = OrderedDict()
metrics["GFLOPS/s"] = lambda p : ((2 * n_columns_A + 1) * n_columns_D * n_rows_D / 1e9) / (p["time"] / 1e3)

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 2 (single precision)...')
results, env = kernel_tuner.tune_kernel(kernel_name, kernel_source, problem_size, arguments,
            tune_params, grid_div_x = grid_div_x, grid_div_y = grid_div_y,
                           verbose=True, metrics=metrics, iterations=32)
print("Done")
print('---------------------------------------------------------------------\n')


# %% Reference Implementation from bvanwekhoven adapted for rectangular matrices (double precision)
# Setup kernel
problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_double, A_double, B_double, C_double, n_columns_A, n_columns_B, n_rows_A]

tune_params = OrderedDict()
tune_params["block_size_x"] = [16*2**i for i in range(3)]
tune_params["block_size_y"] = [2**i for i in range(6)]

tune_params["tile_size_x"] = [2**i for i in range(4)]
tune_params["tile_size_y"] = [2**i for i in range(4)]

grid_div_x = ["block_size_x", "tile_size_x"]
grid_div_y = ["block_size_y", "tile_size_y"]

restrict = ["block_size_x==block_size_y*tile_size_y"]

answer = [numpy.dot(A_double, B_double) * C_double, None, None]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : ((2*n_columns_A + 1)* n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

kernel_name = "matrix_mult_dot_bvanwerkhoven_rectangular_double"
kernel_source = "../cu/matrix_mult_dot_bvanwerkhoven_rectangular_double.cu"

print('\n---------------------------------------------------------------------')
print('Running tiling bvanwerkhoven extension to rectangular matrix (double precision)...')
res, env = kernel_tuner.tune_kernel(kernel_name, kernel_source,
        problem_size, arguments, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x,
        restrictions=restrict, verbose=True, iterations=32, metrics=metrics)

print("Done")
print('---------------------------------------------------------------------\n')



# %% Reference Implementation from bvanwekhoven adapted for rectangular matrices (single precision)
# Setup kernel
problem_size = (n_columns_D, n_rows_D)

arguments = [D_kernel_single, A_single, B_single, C_single, n_columns_A, n_columns_B, n_rows_A]

tune_params = OrderedDict()
tune_params["block_size_x"] = [16*2**i for i in range(3)]
tune_params["block_size_y"] = [2**i for i in range(6)]

tune_params["tile_size_x"] = [2**i for i in range(4)]
tune_params["tile_size_y"] = [2**i for i in range(4)]

grid_div_x = ["block_size_x", "tile_size_x"]
grid_div_y = ["block_size_y", "tile_size_y"]

restrict = ["block_size_x==block_size_y*tile_size_y"]

answer = [numpy.dot(A_single, B_single) * C_single, None, None]

metrics = OrderedDict()
metrics["GFLOP/s"] = lambda p : ((2*n_columns_A + 1) * n_columns_B * n_rows_A/1e9) / (p["time"] / 1e3)

kernel_name = "matrix_mult_dot_bvanwerkhoven_rectangular_single"
kernel_source = "../cu/matrix_mult_dot_bvanwerkhoven_rectangular_single.cu"

print('\n---------------------------------------------------------------------')
print('Running tiling bvanwerkhoven extension to rectangular matrix (single precision)...')
res, env = kernel_tuner.tune_kernel(kernel_name, kernel_source,
        problem_size, arguments, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x,
        restrictions=restrict, verbose=True, iterations=32, metrics=metrics)

print("Done")
print('---------------------------------------------------------------------\n')
