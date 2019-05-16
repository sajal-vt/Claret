from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from pyopencl import device_type
import math
import time
import sys

# define globals
SKIP_LINES = 2
COSCLEN = 51		# length of cosc filter
EPS = 1.0e-5	# termination threshold
cosc = np.array([0.0,-0.00020937301404,-0.00083238644375,-0.00187445134867,-0.003352219513758,-0.005284158713234,-0.007680040381756,-0.010530536243981,-0.013798126870435,-0.017410416484704,-0.021256733995966,-0.025188599234624,-0.029024272810166,-0.032557220569071,-0.035567944643756,-0.037838297355557,-0.039167132882787,-0.039385989227318,-0.038373445436298,-0.036066871845685,-0.032470479106137,-0.027658859359265,-0.02177557557417,-0.015026761314847,-0.007670107630023,0.0,0.007670107630023,0.015026761314847,0.02177557557417,0.027658859359265,0.032470479106137,0.036066871845685,0.038373445436298,0.039385989227318,0.039167132882787,0.037838297355557,0.035567944643756,0.032557220569071,0.029024272810166,0.025188599234624,0.021256733995966,0.017410416484704,0.013798126870435,0.010530536243981,0.007680040381756,0.005284158713234,0.003352219513758,0.00187445134867,0.00083238644375,0.00020937301404,0.0]).astype(float)
g_heir = []	
record_stress = False


def fill_level_count(num_of_points):
	levels = 0
	h = []
	size = num_of_points
	h.append(size)
	levels += 1
	while True:
		if size <= 1000:
			break
		size = math.floor(size / 8)
		h.append(size)
		levels += 1
		
	return levels, h


def loadCSV(filename):
	data = np.array([]).astype(np.float)
	num_of_points = 0
	n_original_dims = 0
	with open(filename, "r") as reader:
		header = reader.readline()
		tokens = header.split(",")
		n_original_dims = len(tokens)

		reader.readline()

		for line in reader:
			tokens = np.array(line.split(",")).astype(np.float)
			data = np.append(data, tokens)
			num_of_points += 1

	return data, num_of_points, n_original_dims

def normalize(data, size, dimension):
	print(np.shape(data))
	data2d = np.reshape(data, (-1, dimension))
	print(np.shape(data2d))
	max_vals = np.amax(data2d, axis = 0)
	min_vals = np.amin(data2d, axis = 0)
	spread_vals = max_vals - min_vals

	for i in range(size):
		for j in range(dimension):
			if spread_vals[j] - min_vals[j] < 0.0001:
				data[i*(dimension)+j] = 0.0
			else:
				data[i*(dimension)+j] = (data[i*(dimension)+j] - min_vals[j]) / spread_vals[j]
				if data[i*(dimension)+j] >= 1000.0 or data[i * dimension + j] <= -1000.0:
					data[i*(dimension)+j] = 0.0



def shuffle(data, size, dimension):
	shuffle_temp = np.ones(dimension)
	shuffle_idx = 0

	for i in range(0, size * dimension, dimension):
		shuffle_idx = int(i + ( np.random.randint(0, 10000) % (size - (i / dimension)) ) * dimension)
		for j in range(dimension):   # swap
			shuffle_temp[j] = data[i + j]
			data[i + j] = data[shuffle_idx + j]
			data[shuffle_idx + j] = shuffle_temp[j]


def distance(data, dimension, i, j):
	point1 = data[i * dimension : (i + 1) * dimension]
	point2 = data[j * dimension : (j + 1) * dimension]

	dist = 0.0
	for i in range(dimension):
		dist += math.pow(point1[i] - point2[i], 2)

	return math.sqrt(dist)


def terminate(iteration, stop_iteration, sstress):
	if iteration - stop_iteration >= 400:
		return True
	signal = 0.0
	if iteration - stop_iteration > COSCLEN:

		for i in range(COSCLEN):
			signal += sstress[ (iteration - COSCLEN)+i ] * cosc[ i ]
		
		if math.fabs( signal ) < EPS:
			return True
	return False

def level_force_directed(
	highD, 
	d_highD,
	lowD,
	d_lowD_a,
	d_lowD_b,
	pivot_indices,
	hd_distances,
	ld_distances,
	d_hd_distances,
	d_ld_distances,
	d_pivot_indices,
	num_of_points,
	n_original_dims,
	n_projection_dims,
	start_index,
	end_index, 
	interpolate, 
	near_set_size,
	random_set_size,
	commands,
	force_kernel,
	ld_kernel,
	stress_kernel,
	resultN,
	resultD,
	d_resultN,
	d_resultD,
	num_of_groups,
	velocity,
	force,
	d_velocity_a,
	d_velocity_b,
	d_force,
	iteration_pp,
	metadata,
	d_metadata,
	d_seed_memory,
	context,
	mf):

	
	if(record_stress):
		stress_writer = open("stress.csv", "w")	

	# Initialize near sets using random values
	modular_operand = 0
	if interpolate == True: 
		modular_operand = start_index
	else:
		modular_operand = end_index

	for i in range(end_index):	
		for j in range(near_set_size):	 
			pivot_indices[i * (near_set_size + random_set_size) + j] = math.floor(np.random.randint(0, 32767) % modular_operand)

	#cl.enqueue_write_buffer(commands, d_pivot_indices, pivot_indices)
	pivot_indices = np.array(pivot_indices).astype(np.uint32)
	d_pivot_indices = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = pivot_indices.nbytes, hostbuf=pivot_indices)


	first_pivots = pivot_indices[0:8]
	print(first_pivots)
	first_highD = highD[first_pivots]

	group_size = 64
	localWorkSize = group_size
	globalWorkSize = int(math.ceil((end_index - start_index) / group_size) * group_size)

	err = 0

	
	sstress = np.zeros(end_index - start_index)

	length = end_index * 8

	

	stress_g_size = int(math.ceil((length / 8.0) / float(group_size)) * group_size)

		 
	for iteration in range(start_index, end_index):
		force_event = force_kernel(commands, (globalWorkSize,), (localWorkSize,), d_highD, d_lowD_a, d_lowD_b, d_velocity_a, d_velocity_b, d_force, d_seed_memory, d_pivot_indices, 
			np.int32(start_index), np.int32(end_index), np.int32(n_original_dims), np.int32(n_projection_dims), np.int32(near_set_size), np.int32(random_set_size), d_hd_distances, d_metadata, np.int32(iteration_pp))

		
		ld_event = ld_kernel(commands, (globalWorkSize,), (localWorkSize,), d_lowD_a, d_lowD_b, d_pivot_indices, d_ld_distances, np.int32(start_index), np.int32(end_index), 
			np.int32(n_projection_dims), np.int32(near_set_size), np.int32(random_set_size), np.int32(iteration_pp))
		              
		
		stress_event = stress_kernel(commands, (stress_g_size,), (localWorkSize,), d_hd_distances, d_ld_distances, cl.LocalMemory(4 * group_size), cl.LocalMemory(4 * group_size), 
			np.int32(length), d_resultN, d_resultD)

		
		resultN_np = np.empty_like(resultN)
		resultD_np = np.empty_like(resultD)

		cl.enqueue_copy(commands, resultN_np, d_resultN)
		cl.enqueue_copy(commands, resultD_np, d_resultD)
	
		

		iteration_pp += 1


		d = 0.0
		n = 0.0
		for k in range(num_of_groups):		
			n += resultN_np[k]
			d += resultD_np[k]

		otherStress = math.sqrt(n / d)
		print("Sress: " + str(otherStress))

		if(record_stress):
			stress_writer.write(str(otherStress) + "\n")
	   
		sstress[iteration - start_index] = otherStress
		if terminate(iteration, start_index, sstress):
			print("Stopping at iteration with stress : " + str(iteration - start_index) + ", " + str(otherStress))
			break
		

	#if(record_stress):
	#	close(stress_writer)
		
	
	
	


def mds(fileName, test_size, record_stress):
	myfile = open(fileName, "r")
	near_set_size = 4
	random_set_size = 4

	n_projection_dims = 2
	highD, num_of_points, n_original_dims = loadCSV(fileName)
	highD = np.array(highD).astype(np.float32)
	print(np.shape(highD))

	group_size = 128

	if(test_size):
		num_of_points = test_size

	num_of_groups = np.int32(math.ceil(num_of_points / float(group_size)))
	velocity = np.array(np.zeros(num_of_points * n_projection_dims)).astype(np.float32)
	force = np.array(np.zeros(num_of_points * n_projection_dims)).astype(np.float32)
	seed_memory = np.array(np.random.randint(2000, size = num_of_points)).astype(np.int32)
	pivot_indices = np.array(np.random.randint(num_of_points, size = num_of_points * (near_set_size + random_set_size))).astype(np.uint32)
	hd_distances = np.array(1.3 * np.ones(num_of_points * (near_set_size + random_set_size))).astype(np.float32)
	ld_distances = np.array(1.3 * np.ones(num_of_points * (near_set_size + random_set_size))).astype(np.float32)
	

	lowD = np.zeros(num_of_points * n_projection_dims)
	for i in range(num_of_points * n_projection_dims):
		lowD[i] = float((np.random.randint(0, 32767) % 10000) / 10000.0)-0.5
	lowD = np.array(lowD).astype(np.float32)


	metadata = np.array(np.zeros(48)).astype(np.float32)
	metadata[30] = 12345.0
	metadata[31] = 12345.0
	metadata[32] = 12345.0
	resultN = np.array(np.zeros(num_of_groups)).astype(np.float32)
	resultD = np.array(np.zeros(num_of_groups)).astype(np.float32)

	metadata[0] = 0 

	print("data size: " + str(num_of_points) + "X" + str(n_original_dims))
	abc = 0	

	spring_force = .7 
	damping = .3
	delta_time = 0.3 
	freeness = .85
	size_factor = 1.0 / (float (near_set_size + random_set_size))

	normalize(highD, num_of_points, n_original_dims)
	shuffle(highD, num_of_points, n_original_dims)
	highD = np.array(highD).astype(np.float32)


	# Boilerplate
	platforms = cl.get_platforms()
	devices = platforms[0].get_devices(cl.device_type.GPU)
	context = cl.create_some_context()
	commands = cl.CommandQueue(context)

	# Load Kernels
	program = cl.Program(context, open("/home/sajal/gitspace/Claret/source/device/claret_kernel.cl").read()).build()
	force_kernel = program.compute_force
	ld_kernel = program.compute_lowdist
	stress_kernel = program.computeStress

	print(str(sys.getsizeof(np.float32(0.0)) * num_of_points * n_original_dims))

	# Create Buffers
	mf = cl.mem_flags
	d_highD =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, size = highD.nbytes, hostbuf=highD)
	d_lowD_a = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = lowD.nbytes, hostbuf=lowD)
	d_lowD_b = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = lowD.nbytes, hostbuf=lowD)
	d_velocity_a = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = velocity.nbytes, hostbuf=velocity)
	d_velocity_b = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = velocity.nbytes, hostbuf=velocity)
	d_force = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = force.nbytes, hostbuf=force)
	d_seed_memory = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = seed_memory.nbytes, hostbuf=seed_memory)
	d_pivot_indices = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = pivot_indices.nbytes, hostbuf=pivot_indices)
	d_hd_distances = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = hd_distances.nbytes, hostbuf=hd_distances)
	d_ld_distances = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = ld_distances.nbytes, hostbuf=ld_distances)
	d_metadata = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = metadata.nbytes, hostbuf=metadata)
	d_resultN = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = resultN.nbytes, hostbuf=resultN)
	d_resultD = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, size = resultD.nbytes, hostbuf=resultD)
	

	# Launch Kernels
	localWorkSize = np.array([32, 0, 0])
	globalWorkSize = np.array([num_of_points, 0, 0])

	iteration = 0
	start_index = 0
	end_index = num_of_points


	print("Initial number of groups: " + str(num_of_groups))




	# Multi-level code
	g_done = False
	g_interpolating = False
	stop_iteration = 0
	g_levels, g_heir = fill_level_count( num_of_points)
	g_current_level = g_levels-1

	iteration_pp = 0
	st = time.time()

	while g_done == False :
		if g_interpolating == True: # interpolate
			print("Interpolating " + str(g_heir[g_current_level]) + " and " + str(g_heir[g_current_level + 1]))
			num_of_groups = math.ceil(g_heir[g_current_level] / float(group_size))
			print("number of groups : " + str(num_of_groups))
			level_force_directed(
					highD, 
					d_highD,
					lowD, 
					d_lowD_a,
					d_lowD_b,
					pivot_indices, 
					hd_distances, 
					ld_distances, 
					d_hd_distances, 
					d_ld_distances, 
					d_pivot_indices, 
					num_of_points,
					n_original_dims, 
					n_projection_dims, 
					g_heir[ g_current_level+1 ], 
					g_heir[g_current_level], 
					g_interpolating,  
					near_set_size,
					random_set_size, 
					commands, 
					force_kernel, 
					ld_kernel, 
					stress_kernel, 
					resultN, 
					resultD, 
					d_resultN, 
					d_resultD, 
					num_of_groups,
					velocity,
					force,
					d_velocity_a,
					d_velocity_b,
					d_force,
					iteration_pp,
					metadata,
					d_metadata,
					d_seed_memory,
					context,
					mf)		  
		else:
			print("Relaxing " + str(g_heir[g_current_level]) + " and 0") 
			num_of_groups = math.ceil(g_heir[g_current_level] / float(group_size))
			print("number of groups : " + str(num_of_groups))	
			level_force_directed(
					highD, 
					d_highD,
					lowD, 
					d_lowD_a,
					d_lowD_b,
					pivot_indices, 
					hd_distances, 
					ld_distances, 
					d_hd_distances, 
					d_ld_distances, 
					d_pivot_indices, 
					num_of_points,
					n_original_dims, 
					n_projection_dims, 
					0, 
					g_heir[g_current_level], 
					g_interpolating,  
					near_set_size,
					random_set_size, 
					commands, 
					force_kernel,  
					ld_kernel, 
					stress_kernel, 
					resultN, 
					resultD, 
					d_resultN, 
					d_resultD, 
					num_of_groups,
					velocity,
					force,
					d_velocity_a,
					d_velocity_b,
					d_force,
					iteration_pp,
					metadata,
					d_metadata,
					d_seed_memory,
					context,
					mf)
		
		if( True ):

			if( g_interpolating ):
				g_interpolating = False
			else:
				g_current_level -= 1 # move to the next level down
				g_interpolating = True

				if( g_current_level < 0 ):
					print("Done")
					g_done = True

		iteration += 1	# increment the current iteration count			
	

	et = time.time()


	print(lowD)
	cl.enqueue_copy(commands, lowD, d_lowD_a)
	cl.enqueue_copy(commands, highD, d_highD)
	print(highD)
	
	print("elapsed time: " + str(et - st))

	projection_writer = open(fileName + ".out.csv", "w")

	for i in range(num_of_points):
		projection_writer.write(str(lowD[i * n_projection_dims + 0]) + "," + str(lowD[i * n_projection_dims + 1]) + "\n")

	projection_writer.close()

np.random.seed(1300)
#mds("/home/sajal/gitspace/Claret/data/breast-cancer-wisconsin.csv", 0, 0)
mds("/home/sajal/gitspace/Claret/data/shuttle_trn_corr.csv", 0, 0)





