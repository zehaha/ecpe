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

# Checks the relationship of the corresponding elements of the two arrays
def check_less(array1, array2):
	count_less = 0
	count_eq = 0
	count_great = 0

	for i in range(0, len(array1)) :
		if(array1[i] <= array2[i]) :
			count_less += 1
			if(array1[i] == array2[i]) :
				count_eq += 1
		else : count_great += 1

	if(count_eq == len(array1)) :
		return 1
	elif(count_great == 0) :
		return 2
	else : return 3

# Merely doing new_array = some_array would just point to, not copy its content
def copy(array1):
	new_array = []
	for i in range(0, len(array1)) :
		new_array.append(array1[i])

	return new_array

# Increment subroutine
def increment(num_array, cand_apps, max_count):
	for i in range(0, len(max_count)):
		if(max_count[i] != 0) :
			temp_array = copy(num_array)
			temp_array[i] += 1
			temp_bool = check_less(temp_array, max_count)

			if(temp_bool == 2) :
				if(temp_array not in cand_apps) :
					cand_apps.append(temp_array)
					increment(temp_array, cand_apps, max_count)
			elif (temp_bool == 1) :
				if(temp_array not in cand_apps) :
					cand_apps.append(temp_array)
				break
			else :
				continue

# Returns an array of 0s and 1s indicating whether the application vector in the ith index is valid or not
def validate(result_matrix, indices, config_vector, candapps, n):
	# for found_platform in cl.get_platforms():
	# 	if found_platform.name == 'NVIDIA CUDA':
	# 		my_platform = found_platform
	device = cl.get_platforms()[1].get_devices()[0]
	ctx = cl.Context([device])


	queue = cl.CommandQueue(ctx,
	        properties=cl.command_queue_properties.PROFILING_ENABLE)

	result_np = np.array(result_matrix, dtype = np.integer).flatten()
	indices_np = np.array(indices, dtype = np.integer)
	init_config_vector_np = np.array(config_vector, dtype=np.integer)
	candapps_np = np.array(candapps, dtype=np.integer)
	result_vector_np = np.empty(n).astype(np.integer)

	kernel = """

		__kernel void validate(__global int* result_vector, __global int* result_matrix, __global int* candapps, 
			__global int* config_vector, __global int* indices){

		int valid = 1;
		int temp_result_matrix[MAT_SIZE];
		int grpid = get_group_id(0);

		if(get_local_id(0) == 0) {
			for(int i = 0; i < MAT_SIZE; i ++) temp_result_matrix[i] = result_matrix[i];

			// Check for validity in general conditions
			for(int i = 0; i < INDICES_LENGTH; i ++) {
				for(int j = 0; j < COL_SIZE; j ++) temp_result_matrix[i * COL_SIZE + j] *= candapps[grpid * COL_SIZE + j];
				int sum = 0;
				for(int j = 0; j < COL_SIZE; j ++) sum += temp_result_matrix[i * COL_SIZE + j];
				if(sum > config_vector[indices[i]]) {
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
							if((sum + 1) > config_vector[indices[j]]) {
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

			if(valid == 1) result_vector[grpid] = 1;
		}
	} 
	"""
	indices_length = "#define INDICES_LENGTH " + str(len(indices)) + '\n'
	mat_size = "#define MAT_SIZE " + str(len(result_np)) + '\n'
	column_size = "#define COL_SIZE " + str(col) + '\n'

	kernel = indices_length + mat_size + column_size + kernel
	program = cl.Program(ctx, kernel).build()

	queue = cl.CommandQueue(ctx)

	# create memory buffers
	mf = cl.mem_flags
	result_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = result_np)
	indices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = indices_np)
	init_config_vector_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = init_config_vector_np)
	candapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = candapps_np)
	result_vector_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result_vector_np.nbytes)

	# execute the kernel
	program.validate(queue, candapps_np.shape, (col, ), result_vector_buf, result_buf, candapps_buf, init_config_vector_buf, indices_buf)

	cl.enqueue_copy(queue, result_vector_np, result_vector_buf)
	return result_vector_np

# Returns the valid application vectors based on the result of validate_apps subroutine
def get_validapps(result_vector_np, candapps, n):
	validapps = []
	for i in range(0, n):
		if result_vector_np[i] == 1:
			for j in range(0, col): validapps.append(candapps[i * col + j])

	return validapps

# Returns the partitioned configuration vectors from the results of compute subroutine
def configvects_partition(result_config_vectors_np, n):
	configvects = []
	for i in range(0, n):
		temp = []
		for j in range(0, row): temp.append(result_config_vectors_np[i * row + j])
		configvects.append(temp)

	return configvects

# Performs the Computation
def compute(trans_matrix, config_vector, validapps, num_valid_apps):
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

		int dot_result[COL_SIZE];
		int grpid = get_group_id(0);

		if(get_local_id(0) == 0) {
			for(int j = 0; j < COL_SIZE; j ++) {
				int sum = 0;
				for(int i = 0; i < ROW_SIZE; i ++) sum += validapps[grpid * ROW_SIZE + i] * trans_matrix[i * COL_SIZE + j];
				dot_result[j] = sum;
			}
			for(int i = 0; i < COL_SIZE; i ++) result_config_vectors[grpid * COL_SIZE + i] = config_vector[i] + dot_result[i];
		}
	} 
	"""
	mat_size = "#define MAT_SIZE " + str(len(trans_np)) + '\n'
	column_size = "#define COL_SIZE " + str(row) + '\n'
	row_size = "#define ROW_SIZE " + str(col) + '\n'

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
	program.compute(queue, validapps_np.shape, (col, ), trans_buf, config_vector_buf, validapps_buf, result_config_vectors_buf)
	cl.enqueue_copy(queue, result_config_vectors_np, result_config_vectors_buf)

	return result_config_vectors_np

time = 0
f = open('../outputs/output_par.txt', 'w')
f.write('')

f = open('../outputs/output_par.txt', 'a')

while True:
	tempstr =  'Time ' + str(time) + ': \n'
	temp_config_set = []
	for config in config_vectors_set :
		tempstr += 'Config vector: ' + str(config) + '\n'
		max_rules = produce_max(config, trigger_matrix)
		cand_apps = []
		# should the user wants to input candidate application vectors manually
		# g = open('cand_apps.txt', 'r')
		# n = int(g.readline())
		# for i in range(0, n):
		# 	cand_apps.append([int(j) for j in g.readline().split(' ')])
		increment([0 for i in range(0, len(max_rules))], cand_apps, max_rules)
		cand_apps = np.array(cand_apps).flatten()
		validapps_indices = validate(result_matrix, indices, config, cand_apps, int(len(cand_apps)/col))
	 	validapps = get_validapps(validapps_indices, cand_apps, int(len(cand_apps)/col))
	 	num_valid_apps = int(len(validapps) / col)
		tempstr += 'Candidate app vectors: ' + str(configvects_partition(cand_apps, num_valid_apps)) + '\n'
		tempstr += 'Applied: ' + str(validapps) + '\n'
		computed_vectors = []
		if len(validapps) != 0: # handles halting configuration
			temp_res_config = compute(trans_matrix, config, validapps, num_valid_apps)
			configvectres = configvects_partition(temp_res_config, num_valid_apps)
			for c in configvectres:
				if c != config: computed_vectors.append(c)
			tempstr += 'Result: ' + str(computed_vectors) + '\n'
			temp_config_set += computed_vectors
			tempstr += ''
	config_vectors_set = [vect for vect in temp_config_set]
	if len(temp_config_set) != 0: config_vectors_set = [temp_config_set[0]]
	else: config_vectors_set = []

	time += 1
	tempstr += ''
	f.write(tempstr)

	if len(config_vectors_set) == 0 : break # all are in non-halting or all are in halting state
f.close()