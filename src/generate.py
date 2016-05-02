from __future__ import division # for automatic precision checking of decimals
import time as myTime # to check for running time of computations
import numpy as np # numpy import
import pyopencl as cl
from pyopencl import array


f = open('../inputs/input.txt', 'r')
temp = f.readline().split()
row = int(temp[0])
col = int(temp[1])

result_matrix = []
trigger_matrix = []
indices = []

f.readline()

# Read and store the resulting matrix of dotting trigger to [1, 1, 1, ... , 1_n] per row, per element
for i in range(0, row):
	temp = [int(j) for j in f.readline().split()]
	if len(temp) != 0 and sum(temp) != 0:
		# for z in temp.split():
		# 	result_matrix.append(z)
		result_matrix.append(temp)
		indices.append(i)
	trigger_matrix.append(temp)

trans_matrix = []

f.readline()

# Read and store the transition matrix
for i in range(0, col):
	temp = [int(j) for j in f.readline().split()]
	if len(temp) != 0:
		trans_matrix.append(temp)

f.readline()

# Read and store the initial configuration vector
config_vector = [int(j) for j in f.readline().split()]
config_vectors_set = []
config_vectors_set.append(config_vector)

f.close()

# Produce maximum number of applications of rules per iteration
def produce_max(config_vector, trigger_matrix):
	max_matrix = []
	config_np = np.array(config_vector)
	trigger_np = np.array(trigger_matrix)
	trigger_np = np.transpose(trigger_np)
	temp_np = np.array(trigger_matrix)
	temp_np = np.transpose(temp_np)

	counter = 0

	for x in range(0, len(trigger_np)):
		while((temp_np[x] <= config_np).all()):
			counter+=1
			temp_np[x] = np.add(temp_np[x], trigger_np[x])
		max_matrix.append(counter)
		counter = 0

	return max_matrix

max_arr = produce_max(config_vector, trigger_matrix)

max_app = max(max_arr)
ker_length = 1
elem_length = [0]*col

for i in reversed(range(col)): 
	ker_length *= (max_arr[i] + 1)
	elem_length[i] = ker_length

#for found_platform in cl.get_platforms():
#	if found_platform.name == 'NVIDIA CUDA': 
#		my_platform = found_platform
#		break

my_platform = cl.get_platforms()[1]
device = my_platform.get_devices()[0]
context = cl.Context([device])

queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

blank_app = [0] * col
apps_np = np.array(blank_app, dtype=np.integer)
result_vector_np = np.empty(ker_length * len(blank_app)).astype(np.integer)
max_arr_np = np.array(max_arr, dtype=np.integer)
elem_length_np = np.array(elem_length, dtype=np.integer)
print max_arr
print elem_length_np
# print ker_length

kernel = """

		__kernel void validate(__global int* result_vector, __global int* elem_length, __global int* max_arr){

		int g_id = get_group_id(0), l_id = get_local_id(0);
		int incr = elem_length[g_id] / (max_arr[g_id] + 1);

		if (l_id <= max_arr[g_id]) {
			for(int i = 0; i < KER_LENGTH; i += elem_length[g_id]) {
				for(int j = 0; j < incr; j++) {
					result_vector[g_id * KER_LENGTH + (incr * l_id) + i + j] = l_id;
				}
			}
		}

	} 
	"""

def_col = "#define COL " + str(col) + "\n"
def_ker_length = "#define KER_LENGTH " + str(ker_length) + "\n"

kernel = def_col + def_ker_length + kernel
program = cl.Program(context, kernel).build()

queue = cl.CommandQueue(context)

# create memory buffers
mf = cl.mem_flags
result_vector_buf = cl.Buffer(context, mf.WRITE_ONLY, result_vector_np.nbytes)
max_arr_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = max_arr_np)
elem_length_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = elem_length_np)

# execute the kernel
complete_events = program.validate(queue, ((max_app+1)*col, ), (max_app+1, ), result_vector_buf, elem_length_buf, max_arr_buf)
events = [complete_events]

# copy back to host
cl.enqueue_copy(queue, result_vector_np, result_vector_buf, wait_for=events)

print result_vector_np
