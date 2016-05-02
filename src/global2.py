from __future__ import division # for automatic precision checking of decimals
import time as myTime # to check for running time of computations
import numpy as np # numpy import
import pyopencl as cl
from pyopencl import array
import sys

f = open('../inputs/1.txt', 'r')
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
	
	my_platform = cl.get_platforms()[1]
	device = my_platform.get_devices()[0]
	context = cl.Context([device])

	queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

	temp_np = np.array(np.transpose(np.array(trigger_matrix)), dtype=np.integer).flatten()
	trigger_np = np.array(np.transpose(np.array(trigger_matrix)), dtype=np.integer).flatten()
	config_vector_np = np.array(config_vector, dtype=np.integer)
	max_np = np.empty(col).astype(np.integer)

	size = sys.getsizeof(temp_np)+sys.getsizeof(trigger_np)+sys.getsizeof(config_vector_np)+sys.getsizeof(max_np)
	if size > device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE): 
		print 'invalid global memory allocation!'
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < col:
		print 'invalid number of work-items!'
		return -1

	kernel = """
		__kernel void produce_max(__global int* temp_vector, __global int* trigger, __global int* config_vector, __global int* max_arr){
			int gid = get_global_id(0), counter, max = 0;

			while(true) {
				counter = 0;
				for(int i = 0; i < ROW; i ++) {
					if(temp_vector[gid * ROW + i] > config_vector[i]) break;
					else counter += 1;
				}
				if(counter == ROW) {
					max += 1;
					for(int i = 0; i < ROW; i ++) temp_vector[gid * ROW + i] += trigger[gid * ROW + i];
				} else break;
			}

			max_arr[gid] = max;
	}
	"""

	def_col = "#define COL " + str(col) + "\n"
	def_row = "#define ROW " + str(row) + "\n"
	kernel = def_col + def_row + kernel

	program = cl.Program(context, kernel).build()
	queue = cl.CommandQueue(context)

	mf = cl.mem_flags
	temp_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = temp_np)
	trigger_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trigger_np)
	config_vector_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = config_vector_np)
	max_buf = cl.Buffer(context, mf.WRITE_ONLY, max_np.nbytes)

	complete_events = program.produce_max(queue, (col, ), None, temp_buf, trigger_buf, config_vector_buf, max_buf)
	events = [complete_events]

	# copy back to host
	cl.enqueue_copy(queue, max_np, max_buf, wait_for=events)

	return list(max_np)

# generate candidate application vectors 
def generate_cand_apps(max_arr):
	max_app = max(max_arr)
	ker_length = 1
	elem_length = [0]*col

	for i in reversed(range(col)): 
		ker_length *= (max_arr[i] + 1)
		elem_length[i] = ker_length

	my_platform = cl.get_platforms()[1]
	device = my_platform.get_devices()[0]
	context = cl.Context([device])

	queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

	apps_np = np.array(blank_app, dtype=np.integer)
	result_vector_np = np.empty((ker_length-1) * col).astype(np.integer)
	max_arr_np = np.array(max_arr, dtype=np.integer)
	elem_length_np = np.array(elem_length, dtype=np.integer)

	size = sys.getsizeof(result_vector_np)+sys.getsizeof(max_arr_np)+sys.getsizeof(elem_length_np)
	if size > device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE): 
		print 'invalid global memory allocation!'
		return -1
	if device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE) < max_app:
		print 'invalid number of work-groups!'
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < max_app:
		print 'invalid number of work-items!'
		return -1

	kernel = """

			__kernel void validate(__global int* result_vector, __global int* elem_length, __global int* max_arr){

			int g_id = get_group_id(0), l_id = get_local_id(0);
			int incr = elem_length[g_id] / (max_arr[g_id] + 1);
			int end = 0;

			if ((l_id + 1) <= max_arr[g_id]) {
				for(int i = 0; i < KER_LENGTH; i += elem_length[g_id]) {
					for(int j = 0; j < incr; j++) result_vector[g_id * KER_LENGTH + (incr * (l_id + 1)) + i + j - (g_id + 1)] = l_id + 1;
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
	complete_events = program.validate(queue, (max_app*col, ), (max_app, ), result_vector_buf, elem_length_buf, max_arr_buf)
	events = [complete_events]

	# copy back to host
	cl.enqueue_copy(queue, result_vector_np, result_vector_buf, wait_for=events)

	return list(result_vector_np)

# get generated candidate application vectors from the result for generate_cand_apps
def get_cand_apps(cand_apps):
	ker_length = int(len(cand_apps) / col)
	cand_apps_arr = []
	for i in range(ker_length):
		temp = [cand_apps[i]]
		for j in range(1, col): temp.append(cand_apps[i + j * (ker_length)])
		cand_apps_arr.append(temp)

	return cand_apps_arr

# validate the generated application vectors
def validate_apps(config_vectors_set, candapps, numapps):
	max_numapps = max(numapps)
	num_configs = len(config_vectors_set)

	device = cl.get_platforms()[1].get_devices()[0]
	ctx = cl.Context([device])

	queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

	result_np = np.array(result_matrix, dtype = np.integer).flatten()
	indices_np = np.array(indices, dtype = np.integer)
	config_vectors_np = np.array(config_vectors_set, dtype=np.integer).flatten()
	candapps_np = np.array(candapps, dtype=np.integer).flatten()
	numapps_np = np.array(numapps, dtype=np.integer)
	result_vector_np = np.empty(sum(numapps)).astype(np.integer)

	size = sys.getsizeof(result_vector_np)+sys.getsizeof(result_np)+sys.getsizeof(candapps_np)+sys.getsizeof(config_vectors_np)+sys.getsizeof(indices_np)
	+sys.getsizeof(numapps_np)
	if size > device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE): 
		print 'invalid global memory allocation!', size, device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE)
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < sum(numapps):
		print 'invalid number of work-items!'
		return -1

	kernel = kernel = """

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

	return result_vector_np

# filter application vectors and configuration vectors based on the result of validate_apps
def filter_apps(config_vectors_set, valid_apps_bool, cand_apps, numapps):
	temp_index_set = set([])
	valid_apps_set = []
	numvalidapps = []


	for i in range(len(valid_apps_bool)):
		if valid_apps_bool[i] != 0 :
			index = 0
			temp = 0
			for j in range(len(config_vectors_set)):
				temp += numapps[j]
				if i < temp:
					index = j
					break

			valid_apps_set.append(cand_apps[i])
			temp_index_set.add(index)

	num_valid_apps = []
	end_index = 0
	start_index = 0
	for i in range(len(config_vectors_set)):
		end_index += numapps[i]
		count_sum = sum(valid_apps_bool[start_index:end_index])
		if count_sum != 0: num_valid_apps.append(count_sum)
		start_index = end_index

	temp_config_vectors_set = []
	for i in temp_index_set: temp_config_vectors_set.append(config_vectors_set[i])

	return valid_apps_set, temp_config_vectors_set, num_valid_apps

# perform the computation: config_vector = config_vector + app_vector * transition matrix
def compute(config_vectors_set, validapps, num_valid_apps):
	max_valid_apps = max(num_valid_apps)
	num_config_vecs = len(config_vectors_set)

	device = cl.get_platforms()[1].get_devices()[0]
	ctx = cl.Context([device])

	platform = cl.get_platforms()[1]
	device = platform.get_devices()[0]

	queue = cl.CommandQueue(ctx,
	        properties=cl.command_queue_properties.PROFILING_ENABLE)

	trans_np = np.array(trans_matrix, dtype = np.integer).flatten()
	config_vector_np = np.array(config_vectors_set, dtype=np.integer).flatten()
	validapps_np = np.array(validapps, dtype=np.integer).flatten()
	num_valid_apps_np = np.array(num_valid_apps, dtype=np.integer)
	result_config_vectors_np = np.empty(sum(num_valid_apps) * row).astype(np.integer)

	size = sys.getsizeof(trans_np)+sys.getsizeof(config_vector_np)+sys.getsizeof(validapps_np)+sys.getsizeof(num_valid_apps_np)
	+sys.getsizeof(result_config_vectors_np)
	if size > device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE): 
		print 'invalid global memory allocation!', size, device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE)
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < sum(num_valid_apps)*row:
		print 'invalid number of work-items!'
		return -1

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

	return list(result_config_vectors_np)

# divides the result of compute into separate arrays
def get_config_vectors(config_vectors):
	start_index = 0
	temp_config_vectors_set = []
	for i in range(int(len(config_vectors) / row)):
		end_index = (i + 1) * row
		temp_config_vectors_set.append(config_vectors[start_index : end_index])
		start_index = end_index

	return temp_config_vectors_set

time = 0

while time != 5:
	tempstr =  'Time ' + str(time) + ':'
	print tempstr
	temp_config_set = []
	numapps = []
	cand_apps = []
	candapps = []
	print 'Config vectors: ' + str(config_vectors_set)
	temp_config_vectors = []
	for config in config_vectors_set :
		max_rules = produce_max(config, trigger_matrix)
		if max_rules == -1 or sum(max_rules) == 0: 
			continue
		else:
			candapps = (get_cand_apps(generate_cand_apps(max_rules)))
			numapps.append(len(candapps))
			cand_apps += candapps
			temp_config_vectors.append(config)

	config_vectors_set = temp_config_vectors
	if len(config_vectors_set) == 0: break
	valid_apps_bool = validate_apps(config_vectors_set, cand_apps, numapps)
	if type(valid_apps_bool) == int: break
	valid_apps, config_vectors_set, numapps = filter_apps(config_vectors_set, valid_apps_bool, cand_apps, numapps)
	print 'Valid app vectors: ' + str(valid_apps)
	config_vectors = compute(config_vectors_set, valid_apps, numapps)
	if config_vectors == -1: break
	temp_res_config = get_config_vectors(config_vectors)
	computed_vectors = []
	for config in temp_res_config:
		if config not in config_vectors_set:
			computed_vectors.append(config)
	temp_config_set += computed_vectors
	config_vectors_set = [vect for vect in temp_config_set]
	print 'Result: ' + str(config_vectors_set) + '\n'

	time += 1
	# tempstr += ''
	if len(config_vectors_set) == 0 : break # all are in non-halting or all are in halting state