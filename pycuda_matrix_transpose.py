# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

import numpy as np
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class Transpose:
    def transpose(self, matrix):
        # a_cpu: a 2D matrix.

        # Host Variables
        self.matrix = matrix
        M, N = self.matrix.shape
        self.T = np.zeros((N, M), dtype=np.float32)

        # Device memory allocation
        self.transpose_d = cuda.mem_alloc(self.T.nbytes)
        self.matrix_d = cuda.mem_alloc(self.matrix.nbytes)

        # kernel
        self.kernel_code_template = """
			#include <stdio.h>

			__global__ void mat_trans(float *T, const float *mat, int P, int L)
			{
                // 2-D thread ID assuming more than one block will be executed
                int index_x = threadIdx.x + blockIdx.x * blockDim.x; // ROWS
                int index_y = threadIdx.y + blockIdx.y * blockDim.y; // COLUMNS

                // index of the input array; L=columns
                int el = index_x * L + index_y;

                // index of the output array (transposed); P = rows
                int out = index_y * P + index_x;
                if(index_x < P && index_y < L){
                    T[out] = mat[el];
                }

            }
		"""
        
        self.kernel_code = self.kernel_code_template % {
        }
        self.mod = SourceModule(self.kernel_code)

        # create CUDA Event to measure time
        start = cuda.Event() #pay attention here: this is the recommended method to record cuda running time
        end = cuda.Event()

		# copy data to device
        cuda.memcpy_htod(self.matrix_d, self.matrix)

        # function call
        func = self.mod.get_function('mat_trans')
        start.record()
        start_ = time.time()
        func(self.transpose_d, self.matrix_d, np.int32(M), np.int32(N), block=(32, 32, 1), grid = (np.int(np.ceil(float(M)/32)), np.int(np.ceil(float(N)/32)),1)) # In CUDA block=(x,y,z), grid=(x,y,z)
        end_ = time.time()
        end.record()

		# memory copy to host
        cuda.memcpy_dtoh(self.T, self.transpose_d)

		# CUDA Event synchronize
        end.synchronize()

        return self.T, start.time_till(end)*1e-3


if __name__ == "__main__":
    iteration = 10
    M = 5
    N = 7
    n = np.arange(0, iteration, 1) # np.arange(start,stop,step)
    py_times = []
    cu_times = []

    for itr in range(iteration):

        # Create the input array
        matrix = np.float32(np.random.randint(low=0, high=255, size=(M*(itr+1),N*(itr+1))))
        
        # Create the output array
        py_output = np.zeros(shape=(N*(itr+1),M*(itr+1)), dtype=np.float32) # Python
        cu_output = None # CUDA

        # Create instance for CUDA
        module = Transpose()

        # Serial (Python)
        times = []
        for e in range(3):
            start = time.time()
            for i in range(matrix.shape[1]): # 0 to N*(itr+1)
                for j in range(matrix.shape[0]): # 0 to M*(itr+1)
                    py_output[i][j]= matrix[j][i]
            times.append(time.time() - start)
        py_times.append(np.average(times))

        # CUDA
        times = []
        for e in range(3):
            cu_output, t = module.transpose(matrix)
            times.append(t)
        cu_times.append(np.average(times))

        print("Code Equality:", np.allclose(py_output, cu_output))
        print("py_time:", py_times[itr])
        print("cu_time:", cu_times[itr])
        print("cu_output:")
        print(cu_output)
        # print("Original:")
        # print(matrix)
        print("py_output")
        print(py_output)
        print()

    # Optional: if you want to plot the function, set MAKE_PLOT to
    # True:
    MAKE_PLOT = True
    if MAKE_PLOT:
        plt.gcf()
        plt.plot((M*n + N*n), py_times,'r', label="Python") # matrix size versus python run times
        plt.plot((M*n + N*n), cu_times,'g', label="CUDA") # matrix size versus CUDA run times
        plt.legend(loc='upper left')
        plt.title('Matrix Transpose')
        plt.xlabel('Matrix Size')
        plt.ylabel('output coding times(sec)')
        plt.gca().set_xlim((min(M*n + N*n), max(M*n + N*n)))
        plt.savefig('plots_pycuda.png')
        