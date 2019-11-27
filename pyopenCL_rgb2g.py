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
import matplotlib.image as mpimg

class RGB2GRAY:
    def rgb2gray(self, rgb_cpu):
        # rgb_cpu: an RGB image
        # return: the grayscale converter image
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
 
		# Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
		
		# host variables
        self.color = rgb_cpu
        m,n,k = self.color.shape
        self.gray = np.zeros(shape=(self.color.shape[0],self.color.shape[1]), dtype=np.float32)
        self.r = np.ascontiguousarray(self.color[:,:,0]) # already dtype=np.float32 since rgb_cpu is rgb from below
        self.g = np.ascontiguousarray(self.color[:,:,1])
        self.b = np.ascontiguousarray(self.color[:,:,2])

		# device memory allocation        
        self.gray_d = cl.array.to_device(self.queue, self.gray)
        self.r_d = cl.array.to_device(self.queue, self.r)
        self.g_d = cl.array.to_device(self.queue, self.g)
        self.b_d = cl.array.to_device(self.queue, self.b)

		# kernel
        self.kernel_code_template = """
			__kernel void rgb_2_g(__global float *gray, __global float *r, __global float *g, __global float *b, int M, int N)
			{
				int id_x = get_global_id(0); // x-direction
                int id_y = get_global_id(1); // y-direction

                // M, N = first 2 dimensions of the color image; index of each r,g,b pixel from color image
                int pixel = id_x * N + id_y;

                float r_element = r[pixel]; // intensity of red of that pixel
                float g_element = g[pixel];
                float b_element = b[pixel];

                gray[pixel] = 0.21 * r_element + 0.71 * g_element + 0.07 * b_element;

			}
		"""

        self.kernel_code = self.kernel_code_template % {
        }
        self.prg = cl.Program(self.ctx, self.kernel_code).build()

        # function call
        func = self.prg.rgb_2_g

        start = time.time()
        evt = func(self.queue, self.r_d.shape, None, self.gray_d.data, self.r_d.data, self.g_d.data, self.b_d.data, np.uint32(m), np.uint32(n))
        evt.wait()
        end = time.time()
        time_ = 1e-9 * (evt.profile.end - evt.profile.start) #this is the recommended way to record OpenCL running time

        # memory copy to host
        self.gray = self.gray_d.get()

        return self.gray, time_

if __name__ == "__main__":
    py_times = []
    cl_times = []

    # Create the input array
    # Change this path to the path of the image after using scp to transfer it the server
    # rgb = mpimg.imread('/Users/DrewAfromsky/Desktop/Fall 2019/EECS 4750- Heterogeneous Comp-Sign Processing/Assignment 2/color.png')
    # rgb = mpimg.imread('/home/daa2162/color.png') # path to png on the server
    rgb = plt.imread('/home/daa2162/color.png') # path to png on the server
    rgb = rgb.astype(np.float32)

    # Create the output array
    py_output = np.zeros(shape=(rgb.shape[0],rgb.shape[1])) # Python
    cl_output = None # OpenCL

    # Create instance for OpenCL
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

    # OpenCL
    times = []
    for e in range(3):
        cl_output, t = module.rgb2gray(rgb)
        times.append(t)
    cl_times.append(np.average(times))

    print("Code Equality:", np.allclose(py_output, cl_output))
    print("OpenCL Times:", cl_times)
    print("Serial Times:", py_times)
    print("Speed-Up OpenCL:", py_times[0]/cl_times[0])

    # # Optional: if you want to plot the function, set MAKE_PLOT to
	# # True:
    MAKE_PLOT = True
    if MAKE_PLOT:
        plt.figure()
        plt.imsave("./gray_scale_OpenCL.png", cl_output, cmap="gray")
        plt.imsave("./py_gray_scale_2.png",py_output, cmap="gray")
