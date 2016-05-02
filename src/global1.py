from __future__ import division # for automatic precision checking of decimals
import time as myTime # to check for running time of computations
import numpy as np # numpy import
import pyopencl as cl
from pyopencl import array
import sys

f = open('../inputs/input.txt', 'r')
temp = f.readline().split()
row = int(temp[0])
col = int(temp[1])

result_matrix = []
trigger_matrix = []
trigger_matrix_ind = []
trigger_matrix_con = []
indices = []

f.readline()

# Read and store the resulting matrix of dotting trigger to [1, 1, 1, ... , 1_n] per row, per element
for i in range(row):
	temp = [int(j) for j in f.readline().split()]
	temp1 = []
	temp2 = []
	if len(temp) != 0 and sum(temp) != 0:
		for k in range(col):
			if temp[k] != 0:
				temp1.append(temp[k])
				temp2.append(i*col+k)
		trigger_matrix_con.append(temp1)
		trigger_matrix_ind.append(temp2)

	if len(temp) != 0 and sum(temp) != 0:
		indices.append(i)
	trigger_matrix.append(temp)

trigger_temp = np.array(np.transpose(trigger_matrix))
result_matrix_ind = []
result_matrix_con = []

for i in range(col): 
	temp1 = []
	temp2 = []
	if len(trigger_temp[i]) != 0 and sum(trigger_temp[i]) != 0:
		for k in range(row):
			if trigger_temp[i][k] != 0:
				temp1.append(trigger_temp[i][k])
				temp2.append(i*row+k)
		result_matrix_con.append(temp1)
		result_matrix_ind.append(temp2)

# print trigger_temp, result_matrix_ind, result_matrix_con

result_matrix_ind = [i for j in result_matrix_ind for i in j]
result_matrix_con = [i for j in result_matrix_con for i in j]

# print trigger_temp, result_matrix_ind, result_matrix_con

trigger_matrix_ind = [i for j in trigger_matrix_ind for i in j]
trigger_matrix_con = [i for j in trigger_matrix_con for i in j]

trans_matrix = []
trans_matrix_ind = []
trans_matrix_con = []

f.readline()

# Read and store the transition matrix
for i in range(col):
	temp = [int(j) for j in f.readline().split()]
	temp1 = []
	temp2 = []
	if len(temp) != 0:
		for k in range(row):
			if temp[k] != 0:
				temp1.append(temp[k])
				temp2.append(i*row+k)
		trans_matrix_con.append(temp1)
		trans_matrix_ind.append(temp2)

	if len(temp) != 0:
		trans_matrix.append(temp)

trans_matrix_ind = [i for j in trans_matrix_ind for i in j]
trans_matrix_con = [i for j in trans_matrix_con for i in j]

# print trigger_matrix, trigger_matrix_ind, trigger_matrix_con
print trans_matrix, trans_matrix_ind, trans_matrix_con

f.readline()

# Read and store the initial configuration vector
config_vector = [int(j) for j in f.readline().split()]
config_vectors_set = []
config_vectors_set.append(config_vector)

f.close()

# Produce maximum number of applications of rules per iteration
def produce_max(config_vector):

	my_platform = cl.get_platforms()[1]
	device = my_platform.get_devices()[0]
	context = cl.Context([device])

	queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

	temp_np = np.array(result_matrix_con, dtype=np.integer)
	trigger_np = np.array(result_matrix_con, dtype=np.integer)
	trigger_ind_np = np.array(result_matrix_ind, dtype=np.integer)
	config_vector_np = np.array(config_vector, dtype=np.integer)
	max_np = np.empty(col).astype(np.integer)

	# print trigger_np, trigger_ind_np

	size = sys.getsizeof(temp_np)+sys.getsizeof(trigger_np)+sys.getsizeof(config_vector_np)+sys.getsizeof(max_np)+sys.getsizeof(trigger_ind_np)
	if size > device.get_info(cl.device_info.GLOBAL_MEM_SIZE): 
		print 'invalid global memory allocation!', size, device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < col:
		print 'invalid number of work-items!'
		return -1

	kernel = """
		__kernel void produce_max(__global int* temp_vector, __global int* trigger, __global int* trigger_ind, __global int* config_vector, __global int* max_arr){
			int gid = get_global_id(0), counter, max = 0, limit = (gid + 1 + (int)(trigger_ind[0] / ROW)) * ROW, start = 0, end = 0;
			int prev = (limit - ROW), with_start = 0;

			// know proper indexing
			for(int i = 0; i < LEN; i ++) {
				if(!(trigger_ind[i] >= prev || trigger_ind[i] < limit)){
					continue;
				}
				if(trigger_ind[i] > limit) {
					end = i;
					break;
				}
				if(i == (LEN - 1)) {
					end = LEN;
				}
				if(with_start == 0 && trigger_ind[i] >= prev) {
					start = i;
					with_start = 1;
				}
			}

			if((end-start) != 0) {
				while(true) {
					counter = 0;
					for(int i = start; i < end; i ++) {
						if(temp_vector[i] > config_vector[trigger_ind[i] % ROW]) break;
						else counter += 1;
					}
					if(counter == (end-start)) {
						max += 1;
						for(int i = start; i < end; i ++) temp_vector[i] += trigger[i];
					} else break;
				}

				max_arr[gid] = max;
			}
	}
	"""
	def_len = "#define LEN " + str(len(temp_np)) + "\n"
	def_col = "#define COL " + str(col) + "\n"
	def_row = "#define ROW " + str(row) + "\n"
	kernel = def_len + def_col + def_row + kernel

	program = cl.Program(context, kernel).build()
	queue = cl.CommandQueue(context)

	mf = cl.mem_flags
	temp_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = temp_np)
	trigger_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trigger_np)
	trigger_ind_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trigger_ind_np)
	config_vector_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = config_vector_np)
	max_buf = cl.Buffer(context, mf.WRITE_ONLY, max_np.nbytes)

	# print sys.getsizeof(temp_buf)

	complete_events = program.produce_max(queue, (col, ), None, temp_buf, trigger_buf, trigger_ind_buf, config_vector_buf, max_buf)
	events = [complete_events]

	# copy back to host
	cl.enqueue_copy(queue, max_np, max_buf, wait_for=events)

	return list(max_np)

max_arr = produce_max(config_vector)
# print max_arr

# generate candidate application vectors 
def generate_cand_apps(max_arr):
	max_arr_ind = []
	max_arr_con = []

	for i in range(len(max_arr)):
		if max_arr[i] != 0:
			max_arr_ind.append(i)
			max_arr_con.append(max_arr[i])

	max_app = max(max_arr)
	ker_length = 1
	elem_length = [0]*len(max_arr_ind)
	arr_len = len(max_arr_ind)

	for i in reversed(range(col)): 
		ker_length *= (max_arr[i] + 1)
		if i in max_arr_ind: 
			arr_len -= 1
			elem_length[arr_len] = ker_length

	my_platform = cl.get_platforms()[1]
	device = my_platform.get_devices()[0]
	context = cl.Context([device])

	queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

	max_arr_len = len(max_arr_ind)
	result_vector_np = np.empty((ker_length-1) * max_arr_len).astype(np.integer)
	max_arr_con_np = np.array(max_arr_con, dtype=np.integer)
	elem_length_np = np.array(elem_length, dtype=np.integer)

	size = sys.getsizeof(result_vector_np)+sys.getsizeof(max_arr_con_np)+sys.getsizeof(elem_length_np)
	if size > device.get_info(cl.device_info.GLOBAL_MEM_CACHE_SIZE): 
		print 'invalid global memory allocation!', size, device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
		return -1
	if device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE) < max_app:
		print 'invalid number of work-groups!'
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < max_app:
		print 'invalid number of work-items!'
		return -1

	kernel = """

			__kernel void validate(__global int* result_vector, __global int* elem_length, __global int* max_arr_con){

			int g_id = get_group_id(0), l_id = get_local_id(0);
			int incr = elem_length[g_id] / (max_arr_con[g_id] + 1);
			int end = 0;

			if ((l_id + 1) <= max_arr_con[g_id]) {
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
	max_arr_con_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = max_arr_con_np)
	elem_length_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = elem_length_np)

	# execute the kernel
	complete_events = program.validate(queue, (max_app*max_arr_len, ), (max_app, ), result_vector_buf, elem_length_buf, max_arr_con_buf)
	events = [complete_events]

	# copy back to host
	cl.enqueue_copy(queue, result_vector_np, result_vector_buf, wait_for=events)

	result_vector = list(result_vector_np)

	for i in range(len(max_arr_ind)-1):
		start = max_arr_ind[i]
		end = max_arr_ind[i+1]
		if((end-start)-1 != 0): 
			for j in range((ker_length-1)*((end-start)-1)):
				result_vector.insert(start+1,0)
	for j in range((ker_length-1)*(col-len(max_arr_ind))):
		result_vector.append(0)

	return result_vector


cand_apps = generate_cand_apps(max_arr)

# get generated candidate application vectors from the result for generate_cand_apps
def get_cand_apps(cand_apps):
	ker_length = int(len(cand_apps) / col)
	cand_apps_arr = []
	for i in range(ker_length):
		temp = [cand_apps[i]]
		for j in range(1, col): temp.append(cand_apps[i + j * (ker_length)])
		cand_apps_arr.append(temp)

	return cand_apps_arr

cand_apps = get_cand_apps(cand_apps)
# print cand_apps
# print trigger_matrix_con, trigger_matrix_ind
numapps = [len(cand_apps)]

# validate the generated application vectors
def validate_apps(config_vectors_set, candapps, numapps):
	# print config_vectors_set, numapps, candapps
	max_numapps = max(numapps)
	num_configs = len(config_vectors_set)

	device = cl.get_platforms()[1].get_devices()[0]
	ctx = cl.Context([device])

	queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

	arr_len = len(trigger_matrix_ind)
	trigger_matrix_ind_np = np.array(trigger_matrix_ind, dtype = np.integer)
	trigger_matrix_con_np = np.array(trigger_matrix_con, dtype = np.integer)
	config_vectors_np = np.array(config_vectors_set, dtype=np.integer).flatten()
	candapps_np = np.array(candapps, dtype=np.integer).flatten()
	numapps_np = np.array(numapps, dtype=np.integer)
	result_vector_np = np.empty(sum(numapps)).astype(np.integer)
	indices_np = np.array(indices, dtype = np.integer)

	size = sys.getsizeof(result_vector_np)+sys.getsizeof(trigger_matrix_ind_np)+sys.getsizeof(candapps_np)+sys.getsizeof(config_vectors_np)+sys.getsizeof(indices_np)
	+sys.getsizeof(numapps_np)+sys.getsizeof(trigger_matrix_con_np)
	if size > device.get_info(cl.device_info.GLOBAL_MEM_SIZE): 
		print 'invalid global memory allocation!', size, device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
		return -1
	if device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE) < num_configs:
		print 'invalid number of work-groups!'
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < max_numapps:
		print 'invalid number of work-items!'
		return -1

	kernel = """

		__kernel void validate(__global int* result_vector, __global int* trigger_matrix_ind, __global int* trigger_matrix_con, __global int* candapps, 
			__global int* config_vectors, __global int* numapps, __global int* indices){

			int valid = 1, temp_result_matrix[LEN], grpid = get_group_id(0), localid = get_local_id(0);
			int first_index = 0;

			__local int config[ROW_SIZE];

			if(localid == 0) {
				for(int i = 0; i < ROW_SIZE; i++) config[i] = config_vectors[grpid * ROW_SIZE + i];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if(grpid > 0) {
				for(int i = 0; i < grpid; i++) first_index += numapps[i];
				first_index *= COL_SIZE;
			}

			if(numapps[grpid] > (int) (localid / ROW_SIZE)) {
				int start = 0, end = 0, limit = 0, prev = 0, with_start;
				for(int i = 0; i < LEN; i ++) temp_result_matrix[i] = trigger_matrix_con[i];

				// Check for validity in general conditions
				for(int i = 0; i < INDICES_LENGTH; i ++) {
					limit = (indices[i] + 1) * COL_SIZE;
					prev = limit - COL_SIZE;
					with_start = 0;
					for(int j = 0; j < LEN; j ++) {
						if(!(trigger_matrix_ind[j] >= prev || trigger_matrix_ind[j] < limit)) continue;
						if(trigger_matrix_ind[j] > limit) {
							end = j;
							break;
						}
						if(j == (LEN - 1)) {
							end = LEN;
						}
						if(with_start == 0 && trigger_matrix_ind[j] >= prev) {
							start = j;
							with_start = 1;
						}
					}
					if((end-start) != 0) {
						for(int j = start; j < end; j ++) temp_result_matrix[j] *= candapps[first_index + (localid * COL_SIZE) + (trigger_matrix_ind[j] % COL_SIZE)];
						int sum = 0;
						for(int j = start; j < end; j ++) sum += temp_result_matrix[j];
						if(sum > config[(int)(trigger_matrix_ind[start] / COL_SIZE)]) {
							valid = 0;
							break;
						}
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
							int index = -1;
							for(int l = 0; l < LEN; l++) {
								if(trigger_matrix_ind[l] == (i + COL_SIZE * indices[j])) {
									index = l;
									break;
								}
							}
							if(index >= 0) {
								limit = ((int)(trigger_matrix_ind[index] / COL_SIZE) + 1) * COL_SIZE;
								prev = limit - COL_SIZE, start = 0, end = 0;
								with_start = 0;
								for(int m = 0; m < LEN; m ++) {
									if(!(trigger_matrix_ind[m] >= prev || trigger_matrix_ind[m] < limit)) continue;
									if(trigger_matrix_ind[m] > limit) {
										end = m;
										break;
									}
									if(m == (LEN - 1)) {
										end = LEN;
									}
									if(with_start == 0 && trigger_matrix_ind[m] >= prev) {
										start = m;
										with_start = 1;
									}
								}
								int sum = 0;
								for(int n = start; n < end; n ++) {
									sum += temp_result_matrix[n];
								}
								if((sum + 1) > config[indices[j]]) {
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
					for(int o = 0; o < COL_SIZE; o ++) sum_set += validity_set[o];
					if(sum_set == truth_count && sum_set != 0) valid = 1;
				}

				if(valid == 1) result_vector[(int)(first_index / COL_SIZE) + localid] = 1;
			}
		} 
		"""
	arr_length = "#define LEN " + str(arr_len) + '\n'
	indices_length = "#define INDICES_LENGTH " + str(len(indices)) + '\n'
	column_size = "#define COL_SIZE " + str(col) + '\n'
	row_size = "#define ROW_SIZE " + str(row) + '\n'

	kernel = arr_length + indices_length + column_size + row_size + kernel
	program = cl.Program(ctx, kernel).build()

	queue = cl.CommandQueue(ctx)

	# create memory buffers
	mf = cl.mem_flags
	trigger_matrix_ind_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trigger_matrix_ind_np)
	trigger_matrix_con_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trigger_matrix_con_np)
	config_vectors_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = config_vectors_np)
	candapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = candapps_np)
	numapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = numapps_np)
	indices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = indices_np)
	result_vector_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result_vector_np.nbytes)

	# execute the kernel
	program.validate(queue, (max_numapps*num_configs, ), (max_numapps, ), result_vector_buf, trigger_matrix_ind_buf, trigger_matrix_con_buf, candapps_buf, config_vectors_buf, numapps_buf, indices_buf)
	cl.enqueue_copy(queue, result_vector_np, result_vector_buf)

	return result_vector_np

valid_apps_bool = list(validate_apps(config_vectors_set, cand_apps, numapps))
# print valid_apps_bool

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

valid_apps, config_vectors_set, num_valid_apps = filter_apps(config_vectors_set, valid_apps_bool, cand_apps, numapps)
# print valid_apps, config_vectors_set, num_valid_apps

# perform the computation: config_vector = config_vector + app_vector * transition matrix
def compute(config_vectors_set, validapps, num_valid_apps):
	max_valid_apps = max(num_valid_apps)
	num_config_vecs = len(config_vectors_set)

	device = cl.get_platforms()[1].get_devices()[0]
	ctx = cl.Context([device])

	platform = cl.get_platforms()[0]
	device = platform.get_devices()[0]

	queue = cl.CommandQueue(ctx,
	        properties=cl.command_queue_properties.PROFILING_ENABLE)

	arr_len = len(trans_matrix_ind)
	trans_matrix_ind_np = np.array(trans_matrix_ind, dtype = np.integer)
	trans_matrix_con_np = np.array(trans_matrix_con, dtype = np.integer)
	config_vector_np = np.array(config_vectors_set, dtype=np.integer).flatten()
	validapps_np = np.array(validapps, dtype=np.integer).flatten()
	num_valid_apps_np = np.array(num_valid_apps, dtype=np.integer)
	result_config_vectors_np = np.empty(sum(num_valid_apps) * row).astype(np.integer)
	# print max_valid_apps*row

	size = sys.getsizeof(trans_matrix_ind_np)+sys.getsizeof(trans_matrix_con_np)+sys.getsizeof(config_vector_np)+sys.getsizeof(validapps_np)+sys.getsizeof(num_valid_apps_np)
	+sys.getsizeof(result_config_vectors_np)
	
	if size > device.get_info(cl.device_info.GLOBAL_MEM_SIZE): 
		print 'invalid global memory allocation!', size, device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
		return -1
	if device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE) < num_config_vecs:
		print 'invalid number of work-groups!'
		return -1
	if device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0] < max_valid_apps*row:
		print 'invalid number of work-items!'
		return -1

	kernel = """

		__kernel void compute(__global int* trans_matrix_con, __global int* trans_matrix_ind, __global int* config_vector, __global int* validapps, __global int* numvalidapps,
			__global int* result_config_vectors){

		int grpid = get_group_id(0), localid = get_local_id(0), sum = 0, first_index = 0;
		__local int config[ROW];

		if(localid == 0) {
			for(int i = 0; i < ROW; i++) config[i] = config_vector[grpid * ROW + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(grpid > 0) {
			for(int i = 0; i < grpid; i++) first_index += numvalidapps[i];
			first_index *= COL;
		}

		if(numvalidapps[grpid] > (int) (localid / ROW)) {
			for(int i = 0; i < COL; i ++) {
				for(int j = 0; j < LEN; j ++) {
					if((localid % ROW) + (ROW * i) == trans_matrix_ind[j]) {
						sum += trans_matrix_con[j] * validapps[first_index + ((int)(localid / ROW) * COL) + (int)(trans_matrix_ind[j] / ROW)];
						break;
					}
				}
			}
			result_config_vectors[(int) (first_index * ROW / COL) + localid] = config[((int) (localid % ROW))] + sum;
		}

	} 
	"""
	arr_length = "#define LEN " + str(arr_len) + '\n'
	column_size = "#define COL " + str(col) + '\n'
	row_size = "#define ROW " + str(row) + '\n'

	kernel = arr_length + column_size + row_size + kernel
	program = cl.Program(ctx, kernel).build()

	queue = cl.CommandQueue(ctx)

	# create memory buffers
	mf = cl.mem_flags
	trans_matrix_ind_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trans_matrix_ind_np)
	trans_matrix_con_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = trans_matrix_con_np)
	config_vector_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = config_vector_np)
	validapps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = validapps_np)
	num_valid_apps_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = num_valid_apps_np)
	result_config_vectors_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result_config_vectors_np.nbytes)

	# execute the kernel
	# 1 work-group = 1 configuration vector = (max num. of valid apps * row) work items
	# 1 work-item = 1 column of transition matrix

	program.compute(queue, (max_valid_apps*row*num_config_vecs, ), (max_valid_apps*row, ), trans_matrix_con_buf, trans_matrix_ind_buf, config_vector_buf, validapps_buf, num_valid_apps_buf, result_config_vectors_buf)
	cl.enqueue_copy(queue, result_config_vectors_np, result_config_vectors_buf)

	return list(result_config_vectors_np)

config_vectors = compute(config_vectors_set, valid_apps, num_valid_apps)

# divides the result of compute into separate arrays
def get_config_vectors(config_vectors):
	start_index = 0
	temp_config_vectors_set = []
	for i in range(int(len(config_vectors) / row)):
		end_index = (i + 1) * row
		temp_config_vectors_set.append(config_vectors[start_index : end_index])
		start_index = end_index

	return temp_config_vectors_set

print get_config_vectors(config_vectors)

time = 0

# while time != 4:
# 	tempstr =  'Time ' + str(time) + ':'
# 	print tempstr
# 	temp_config_set = []
# 	numapps = []
# 	cand_apps = []
# 	candapps = []
# 	print 'Config vectors: ' + str(config_vectors_set)
# 	temp_config_vectors = []
# 	for config in config_vectors_set :
# 		max_rules = produce_max(config, trigger_matrix)
# 		if max_rules == -1 or sum(max_rules) == 0: 
# 			continue
# 		else:
# 			candapps = (get_cand_apps(generate_cand_apps(max_rules)))
# 			numapps.append(len(candapps))
# 			cand_apps += candapps
# 			temp_config_vectors.append(config)

# 	config_vectors_set = temp_config_vectors
# 	if len(config_vectors_set) == 0: break
# 	valid_apps_bool = validate_apps(config_vectors_set, cand_apps, numapps)
# 	if type(valid_apps_bool) == int: break
# 	valid_apps, config_vectors_set, numapps = filter_apps(config_vectors_set, valid_apps_bool, cand_apps, numapps)
# 	print 'Valid app vectors: ' + str(valid_apps)
# 	config_vectors = compute(config_vectors_set, valid_apps, numapps)
# 	if config_vectors == -1: break
# 	temp_res_config = get_config_vectors(config_vectors)
# 	computed_vectors = []
# 	for config in temp_res_config:
# 		if config not in config_vectors_set:
# 			computed_vectors.append(config)
# 	temp_config_set += computed_vectors
# 	config_vectors_set = [vect for vect in temp_config_set]
# 	print 'Result: ' + str(config_vectors_set) + '\n'

# 	time += 1
# 	# tempstr += ''
# 	if len(config_vectors_set) == 0 : break # all are in non-halting or all are in halting state