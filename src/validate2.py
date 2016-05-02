from __future__ import division # for automatic precision checking of decimals
import time as myTime # to check for running time of computations
import numpy as np # numpy import
import pyopencl as cl
from pyopencl import array

trigger_matrix = [[0, 0, 0, 0],
				[1, 1, 1, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 1],
				[0, 0, 0, 0]]

result_matrix = []
indices = []

row = len(trigger_matrix)
col = len(trigger_matrix[0])

for i in range(len(trigger_matrix)):
	if sum(trigger_matrix[i]) != 0:
		result_matrix.append(trigger_matrix[i])
		indices.append(i)
				
config_vectors = [2,1,0,0,0,0,1,2,0,0,0,0,2,1,0,0,0,0]
candapps = [1,0,0,0,0,1,0,0,1,1,1,1,1,2,0,0,1,1,0,0,1,0,0,0]
numapps = [3,2,1]
max_numapps = max(numapps)
num_configs = int(len(config_vectors) / row)

device = cl.get_platforms()[1].get_devices()[0]
ctx = cl.Context([device])

queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

result_np = np.array(result_matrix, dtype = np.integer).flatten()
indices_np = np.array(indices, dtype = np.integer)
config_vectors_np = np.array(config_vectors, dtype=np.integer)
candapps_np = np.array(candapps, dtype=np.integer)
numapps_np = np.array(numapps, dtype=np.integer)
result_vector_np = np.empty(sum(numapps)).astype(np.integer)

print result_matrix, len(result_np), candapps

kernel = """

	__kernel void validate(__global int* result_vector, __global int* result_matrix, __global int* candapps, 
		__global int* config_vectors, __global int* indices, __global int* numapps){

		int valid = 1, temp_result_matrix[MAT_SIZE], glblid = get_global_id(0);
		int first_index = 0, config_index = 0, temp = 0;

		// find which config vector to use
		for(int i = 0; i < CONFIGS; i++) {
			temp += numapps[i];
			if((int) (glblid) < temp) {
				config_index = i;
				break;
		}
	}

		for(int i = 0; i < MAT_SIZE; i ++) temp_result_matrix[i] = result_matrix[i];

		// Check for validity in general conditions
		for(int i = 0; i < INDICES_LENGTH; i ++) {
			for(int j = 0; j < COL_SIZE; j ++) temp_result_matrix[i * COL_SIZE + j] *= candapps[glblid * COL_SIZE + j];
			int sum = 0;
			for(int j = 0; j < COL_SIZE; j ++) sum += temp_result_matrix[i * COL_SIZE + j];
			if(sum > config_vectors[config_index * ROW_SIZE + indices[i]]) {
				valid = 0;
				break;
			}
		}

		// Check for additional required conditions per rule
		if(valid == 1) {
			valid = 0;
			int validity_set[COL_SIZE];
			for(int i = 0; i < COL_SIZE; i ++) validity_set[i] = 0;
			int truth_count = 0;

			int i = 0;
			while(i < COL_SIZE) {
				int j = 0;
				while(j < INDICES_LENGTH) {
					if(temp_result_matrix[j * INDICES_LENGTH + i] != 0) {
						int sum = 0;
						for(int k = 0; k < COL_SIZE; k ++) sum += temp_result_matrix[j * COL_SIZE + k];
						if((sum + 1) > config_vectors[config_index * ROW_SIZE + indices[j]]) {
							validity_set[i] = 1;
							truth_count += 1;
							break;
						}
					}
					j ++;
				}
				i ++;
			}
			int sum_set = 0;
			for(int i = 0; i < COL_SIZE; i ++) sum_set += validity_set[i];
			if(sum_set == truth_count && sum_set != 0) valid = 1;
		}
		if(valid == 1) result_vector[glblid] = 1;
	} 
	"""
indices_length = "#define INDICES_LENGTH " + str(len(indices)) + '\n'
mat_size = "#define MAT_SIZE " + str(len(result_np)) + '\n'
column_size = "#define COL_SIZE " + str(col) + '\n'
row_size = "#define ROW_SIZE " + str(row) + '\n'

kernel = indices_length + mat_size + column_size + row_size + "#define CONFIGS " + str(num_configs) + kernel
program = cl.Program(ctx, kernel).build()

queue = cl.CommandQueue(ctx)

# create memory buffers
mf = cl.mem_flags
result_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = result_np)
indices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = indices_np)
config_vectors_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = config_vectors_np)
candapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = candapps_np)
numapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = numapps_np)
result_vector_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result_vector_np.nbytes)

# execute the kernel
program.validate(queue, (sum(numapps), ), None, result_vector_buf, result_buf, candapps_buf, config_vectors_buf, indices_buf, numapps_buf)

cl.enqueue_copy(queue, result_vector_np, result_vector_buf)

print result_vector_np, max_numapps*num_configs