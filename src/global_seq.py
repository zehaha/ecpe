import numpy as np
import time as myTime

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

# Multiplies the corresponding elements of two arrays
def mult_arrs(array1, array2) :
	length = len(array1)
	res_array = [0 for i in range(0, length)]

	for i in range(0, length) :
		res_array[i] = array1[i] * array2[i]

	return res_array

# Gets the sum of all elements of an array
def get_sum(array1) :
	total = 0
	for i in range(0, len(array1)) : total += array1[i]
	return total

# Gets the dot product of a vector and a matrix
def get_dot(vect, matrix) :
	rows = len(matrix)
	cols = len(matrix[0])

	res = [0 for i in range(0, cols)]

	for i in range(0, cols) :
		tot = 0
		for j in range(0, rows) :
			tot += vect[j] * matrix[j][i]
		res[i] = tot
	return res

# Gets the resulting vector as sum of two vectors
def get_sum_vect(vect1, vect2) :
	length = len(vect1)
	res = [0 for i in range(0, length)]

	for i in range(0, length) :
		res[i] = vect1[i] + vect2[i]

	return res

# Returns an array of 0s and 1s indicating whether the application vector in the ith index is valid or not
def validate_apps(config_vector, cand_apps, result_matrix, indices):
	valid_apps = []
	for s in range(0, len(cand_apps)): valid_apps.append(0)
	for z in range (0, len(cand_apps)):
		valid = True
		temp_result_matrix = []
		for i in result_matrix: temp_result_matrix.append(i)

		# Check for validity in general conditions
		for i in range(0, len(indices)):
			temp_result_matrix[i] = mult_arrs(result_matrix[i], cand_apps[z])
			if(get_sum(temp_result_matrix[i]) > config_vector[indices[i]]):
				valid = False
				break

		# Check for additional required conditions per rule
		if valid:
			i = 0
			valid = False
			validity_set = []
			for x in range(0, col): validity_set.append(0)

			truth_count = 0
			while(i < col) : 
				j = 0
				while(j < len(indices)) : 
					if(temp_result_matrix[j][i] != 0) :
						if((get_sum(temp_result_matrix[j]) + 1) > config_vector[indices[j]]) : 
							validity_set[i] = 1
							truth_count += 1
							break
					j += 1
				i += 1

			sum_set = get_sum(validity_set)
			if sum_set == truth_count and sum_set != 0:
				valid = True

		if valid:
			valid_apps[z] = 1;

	return valid_apps

# Returns the valid application vectors based on the result of validate_apps subroutine
def get_validapps(result_vector_np, candapps, n):
	validapps = []
	for i in range(0, n):
		if result_vector_np[i] == 1:
			validapps.append(candapps[i])

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
def compute(app_vectors, config_vector, trans_matrix):
	n = len(app_vectors)
	result_configvects = []
	for app_vector in app_vectors:
		for k in get_sum_vect(config_vector, get_dot(app_vector, trans_matrix)): result_configvects.append(k)

	return result_configvects

# Produces maximum number of applications of rules per iteration
def produce_max(config_vector, trigger_matrix):
	max_matrix = []
	config_np = np.array(config_vector)
	trigger_np = np.array(trigger_matrix)
	trigger_np = np.transpose(trigger_np)
	temp_np = np.array(trigger_matrix)
	temp_np = np.transpose(temp_np)

	counter = 0
	for x in range(0, len(temp_np)):
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

# Simulation
time = 0
f = open('../outputs/output_seq.txt', 'w')
f.write('')

f = open('../outputs/output_seq.txt', 'a')
while True:
	tempstr =  'Time ' + str(time) + ': \n'
	temp_config_set = []
	for config in config_vectors_set :
		tempstr += 'Config vector: ' + str(config) + '\n'
		maxarr = produce_max(config, trigger_matrix)
		cand_apps = []
		# should the user wants to input candidate application vectors manually
		# g = open('cand_apps.txt', 'r')
		# n = int(g.readline())
		# for i in range(0, n):
		# 	cand_apps.append([int(j) for j in g.readline().split(' ')])
		num_array = [0 for i in range(col)]
		increment(num_array, cand_apps, maxarr)
		validapps_indices = validate_apps(config, cand_apps, result_matrix, indices)
		validapps = get_validapps(validapps_indices, cand_apps, len(cand_apps))
		num_valid_apps = len(validapps)
		tempstr += 'Candidate app vectors: ' + str(cand_apps) + '\n'
		tempstr += 'Applied: ' + str(validapps) + '\n'
		computed_vectors = []
		if len(validapps) != 0: # handles halting configuration
			temp_res_config = compute(validapps, config, trans_matrix)
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