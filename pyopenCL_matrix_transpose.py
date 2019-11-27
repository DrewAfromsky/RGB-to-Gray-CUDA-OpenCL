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
import pyopencl as cl
import pyopencl.array

class Transpose:
    def transpose(self, matrix):
        # a_cpu: a 2D matrix.
        # return: the transpose of a_cpu
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Host Variables
        self.matrix = matrix
        M, N = self.matrix.shape
        self.T = np.zeros((N, M), dtype=np.float32)

        # Device memory allocation
        self.transpose_d = cl.array.to_device(self.queue, self.T) 
        self.matrix_d = cl.array.to_device(self.queue, self.matrix)

        # kernel
        self.kernel_code_template = """
			__kernel void mat_trans(__global float *T, __global const float *mat, int P, int L)
			{
                // 2-D thread ID assuming more than one block will be executed
                int index_x = get_global_id(0); // ROWS
                int index_y = get_global_id(1); // COLUMNS

                // index of the input array; L=columns
                int el = index_x * L + index_y;

                // index of the output array (transposed); P = rows
                int out = index_y * P + index_x;

                if(index_x < P && index_y < L)
                {
                    T[out] = mat[el];
                }

			}
		"""
 
        self.kernel_code = self.kernel_code_template % {
        }
        self.prg = cl.Program(self.ctx, self.kernel_code).build()

        # function call
        func = self.prg.mat_trans

        start = time.time()
        evt = func(self.queue, self.matrix.shape, None, self.transpose_d.data, self.matrix_d.data, np.uint32(M), np.uint32(N))
        evt.wait()
        end = time.time()
        time_ = 1e-9 * (evt.profile.end - evt.profile.start) #this is the recommended way to record OpenCL running time 

        # memory copy to host
        self.T = self.transpose_d.get()

        return self.T, time_

if __name__ == "__main__":
    iteration = 10
    M = 5
    N = 7
    n = np.arange(0, iteration, 1) # np.arange(start,stop,step)
    py_times = []
    cl_times = []

    for itr in range(iteration):

        # Create the input array
        matrix = np.float32(np.random.randint(low=0, high=255, size=(M*(itr+1),N*(itr+1))))
        
        # Create the output array
        py_output = np.zeros(shape=(N*(itr+1),M*(itr+1)), dtype=np.float32) # Python
        cl_output = None # OpenCL

        # Create instance for OpenCL
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

        # OpenCL
        times = []
        for e in range(3):
            cl_output, t = module.transpose(matrix)
            times.append(t)
        cl_times.append(np.average(times))

        print("Code Equality:", np.allclose(py_output, cl_output))
        print("py_time:", py_times[itr])
        print("cl_time:", cl_times[itr])
        print("cu_output:")
        print(cl_output)
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
        plt.plot((M*n + N*n), cl_times,'g', label="OpenCL") # matrix size versus CUDA run times
        plt.legend(loc='upper left')
        plt.title('Matrix Transpose')
        plt.xlabel('Matrix Size')
        plt.ylabel('output coding times(sec)')
        plt.gca().set_xlim((min(M*n + N*n), max(M*n + N*n)))
        plt.savefig('plots_pyOpenCL.png')
        

























