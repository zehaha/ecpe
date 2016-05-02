from __future__ import division # for automatic precision checking of decimals
import time as myTime # to check for running time of computations
import numpy as np # numpy import
import pyopencl as cl
from pyopencl import array

trans_matrix = [[2,1,0,0,0,0],
				[0,-1,1,0,0,0],
				[0,-1,-1,0,1,0],
				[0,0,0,0,0,0]]

				# 2 1 0 0 0 0 -> 4 2 0 0 0 0
				# 0 -1 1 0 0 0 -> 2 0 1 0 0 0
				# 2 -1 0 0 1 0 -> 4 0 0 0 1 0
				# 2 -1 2 0 0 0 -> 3 1 2 0 0 0
				# 2 0 1 0 0 0 -> 3 2 1 0 0 0
				
config_vector = [2,1,0,0,0,0,1,2,0,0,0,0,2,1,0,0,0,0]
validapps = [1,0,0,0,0,1,0,0,1,1,1,1,1,2,0,0,1,1,0,0,1,0,0,0]
num_valid_apps = [3,2,1]
max_valid_apps = max(num_valid_apps)
		#	 0 1 2 3 0 1 2 3
		#	 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
row = 6
col = 4

num_config_vecs = int(len(config_vector) / row)

# computation
device = cl.get_platforms()[1].get_devices()[0]
# print device.max_work_item_sizes
ctx = cl.Context([device])

platform = cl.get_platforms()[1]
device = platform.get_devices()[0]

queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

trans_np = np.array(trans_matrix, dtype = np.integer).flatten()
config_vector_np = np.array(config_vector, dtype=np.integer)
validapps_np = np.array(validapps, dtype=np.integer)
num_valid_apps_np = np.array(num_valid_apps, dtype=np.integer)
result_config_vectors_np = np.empty(sum(num_valid_apps) * row).astype(np.integer)

kernel = """

	__kernel void compute(__global int* trans_matrix, __global int* config_vector, __global int* validapps, __global int* numvalidapps,
		__global int* result_config_vectors){

	int glblid = get_global_id(0), sum = 0, config_index = 0, temp = 0;

	// find which config vector to use
	for(int i = 0; i < CONFIGS; i++) {
		temp += numvalidapps[i];
		if((int) (glblid / ROW) < temp) {
			config_index = i;
			break;
		}
	}

	for(int i = 0; i < COL; i++) sum += trans_matrix[glblid % ROW + (i * ROW)] * validapps[((int) (glblid / ROW)) * COL + i];
	result_config_vectors[glblid] = config_vector[config_index * ROW + (glblid % ROW)] + sum;

} 
"""
mat_size = "#define MAT_SIZE " + str(len(trans_np)) + '\n'
column_size = "#define COL " + str(col) + '\n'
row_size = "#define ROW " + str(row) + '\n'

kernel = mat_size + column_size + row_size + "#define CONFIGS " + str(num_config_vecs) + '\n' + kernel
program = cl.Program(ctx, kernel).build()

queue = cl.CommandQueue(ctx)

# create memory buffers
mf = cl.mem_flags
trans_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trans_np)
config_vector_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = config_vector_np)
validapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = validapps_np)
num_valid_apps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = num_valid_apps_np)
result_config_vectors_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result_config_vectors_np.nbytes)

# execute the kernel
# 1 work-item = 1 column of transition matrix = 1 index of configuration vectors
program.compute(queue, (sum(num_valid_apps)*row, ), None, trans_buf, config_vector_buf, validapps_buf, num_valid_apps_buf, result_config_vectors_buf)
cl.enqueue_copy(queue, result_config_vectors_np, result_config_vectors_buf)

print result_config_vectors_np