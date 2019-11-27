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
import matplotlib.image as mpimg
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class RGB2GRAY:
    def rgb2gray(self, rgb_cpu):
        # rgb_cpu: an RGB image
        
        # Host variables
        self.color = rgb_cpu
        m,n,k = self.color.shape
        self.gray = np.zeros(shape=(self.color.shape[0],self.color.shape[1]), dtype=np.float32)
        self.r = np.ascontiguousarray(self.color[:,:,0]) # already dtype=np.float32 since rgb_cpu is rgb from below
        self.g = np.ascontiguousarray(self.color[:,:,1])
        self.b = np.ascontiguousarray(self.color[:,:,2])

        # Device memory allocation
        self.gray_d = cuda.mem_alloc(self.gray.nbytes)
        self.r_d = cuda.mem_alloc(self.r.nbytes)
        self.g_d = cuda.mem_alloc(self.g.nbytes)
        self.b_d = cuda.mem_alloc(self.b.nbytes)

        # copy data to device
        cuda.memcpy_htod(self.r_d, self.r)
        cuda.memcpy_htod(self.g_d, self.g)
        cuda.memcpy_htod(self.b_d, self.b)

        # kernel
        self.kernel_code_template = """
			#include <stdio.h>

			__global__ void rgb_2_g(float *gray, float *r, float *g, float *b, int M, int N)
			{
                // 2-D thread ID assuming more than one block will be executed
                int index_x = threadIdx.x + blockIdx.x * blockDim.x; // ROWS
                int index_y = threadIdx.y + blockIdx.y * blockDim.y; // COLUMNS

                // M, N = first 2 dimensions of the color image; index of each r,g,b pixel from color image
                int pixel = index_x * N + index_y;

                float r_element = r[pixel]; // intensity of red of that pixel
                float g_element = g[pixel];
                float b_element = b[pixel];

                gray[pixel] = 0.21 * r_element + 0.71 * g_element + 0.07 * b_element;

            }
		"""
        self.kernel_code = self.kernel_code_template % {
        }
        self.mod = SourceModule(self.kernel_code)

        # create CUDA Event to measure time
        start = cuda.Event() #pay attention here: this is the recommended method to record cuda running time
        end = cuda.Event()

        # function call
        func = self.mod.get_function('rgb_2_g')
        start.record()
        start_ = time.time()
        # in CUDA block=(x,y,z), grid=(x,y,z)
        # maximum number of threads single block can have 
        func(self.gray_d, self.r_d, self.g_d, self.b_d, np.int32(m), np.int32(n), block=(32, 32, 1), grid = (np.int(np.ceil(float(m)/32)), np.int(np.ceil(float(n)/32)),1)) # In CUDA block=(x,y,z), grid=(x,y,z)
        end_ = time.time()
        end.record()

        # memory copy to host
        cuda.memcpy_dtoh(self.gray, self.gray_d)

        # CUDA Event synchronize
        end.synchronize()

        # return: the grayscale converter image
        return self.gray, start.time_till(end)*1e-3

if __name__ == "__main__":
    py_times = []
    cu_times = []

    # Create the input array
    # Change this path to the path of the image after using scp to transfer it the server
    # rgb = mpimg.imread('/Users/DrewAfromsky/Desktop/Fall 2019/EECS 4750- Heterogeneous Comp-Sign Processing/Assignment 2/color.png')
    # rgb = mpimg.imread('/home/daa2162/color.png') # path to png on the server
    rgb = plt.imread('/home/daa2162/color.png') # path to png on the server
    rgb = rgb.astype(np.float32)
        
    # Create the output array
    py_output = np.zeros(shape=(rgb.shape[0],rgb.shape[1])) # Python
    cu_output = None # CUDA

    # Create instance for CUDA
    module = RGB2GRAY()

    # Serial (Python)
    times = []
    for e in range(3):
        start = time.time()
        for i in range(len(rgb)): # 0 to 383
            for j in range(len(rgb[i])): # 0 to 511
                grayscale = 0.21 * rgb[i][j][0] + 0.71 * rgb[i][j][1] + 0.07 * rgb[i][j][2]
                py_output[i][j] = grayscale
        times.append(time.time() - start)
        # Display and save the figure 
        # plt.imshow(py_output, cmap = plt.get_cmap('gray'))
        # plt.savefig('/Users/DrewAfromsky/Desktop/Fall 2019/EECS 4750- Heterogeneous Comp-Sign Processing/Assignment 2/serial_grayscale.png')
    py_times.append(np.average(times))

    # CUDA
    times = []
    for e in range(3):
        cu_output, t = module.rgb2gray(rgb)
        times.append(t)
    cu_times.append(np.average(times))

    print("Code Equality:", np.allclose(py_output, cu_output))
    print("CUDA Times:", cu_times)
    print("Serial Times:", py_times)
    print("Speed-Up CUDA:", py_times[0]/cu_times[0])

	# Optional: if you want to plot the function, set MAKE_PLOT to
	# True:
    MAKE_PLOT = True
    if MAKE_PLOT:
        plt.figure()
        plt.imsave("./gray_scale_CUDA.png", cu_output, cmap="gray")
        plt.imsave("./py_gray_scale.png",py_output, cmap="gray")
