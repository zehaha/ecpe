from __future__ import division # for automatic precision checking of decimals
import time as myTime # to check for running time of computations
import numpy as np # numpy import
import pyopencl as cl
from pyopencl import array

trans_matrix = [[2,1,0,0,0,0], [0,-1,1,0,0,0], [0,-1,-1,0,1,0], [0,0,0,0,0,0]]
config_vector = [2,1,0,0,0,0,2,1,0,0,0,0]
validapps = [1,0,0,0,0,1,0,0]
num_valid_apps = 2
app_per_con = 1
row = 6
col = 4

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
result_config_vectors_np = np.empty(num_valid_apps * row).astype(np.integer)

kernel = """

	__kernel void compute(__global int* trans_matrix, __global int* config_vector, __global int* validapps, __global int* result_config_vectors){

	int grpid = get_group_id(0);
	int lclid = get_local_id(0);
	int tobeAdded = 0;
	for (int b=0; b<ROW; b++){
		for(int a=0; a<COL; a++){
			tobeAdded += validapps[a + ((grpid * COL * APP_PER_CONF) + lclid * COL)] * trans_matrix[b + (a * ROW)];
		}
		result_config_vectors[((grpid * ROW * APP_PER_CONF) + lclid * ROW) + b] = tobeAdded + config_vector[((grpid * ROW * APP_PER_CONF) + lclid * ROW) + b];
		tobeAdded = 0;
	}

} 
"""
mat_size = "#define APP_PER_CONF " + str(app_per_con) + '\n'
column_size = "#define COL " + str(col) + '\n'
row_size = "#define ROW " + str(row) + '\n'

kernel = mat_size + column_size + row_size + kernel
program = cl.Program(ctx, kernel).build()

queue = cl.CommandQueue(ctx)

# create memory buffers
mf = cl.mem_flags
trans_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trans_np)
config_vector_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = config_vector_np)
validapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = validapps_np)
result_config_vectors_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result_config_vectors_np.nbytes)



# execute the kernel
program.compute(queue, np.zeros((num_valid_apps, )).shape, np.zeros((app_per_con, )).shape, trans_buf, config_vector_buf, validapps_buf, result_config_vectors_buf)
cl.enqueue_copy(queue, result_config_vectors_np, result_config_vectors_buf)

print np.zeros((num_valid_apps, )).shape, np.zeros((app_per_con, )).shape#, result_config_vectors_np