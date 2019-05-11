from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from pyopencl import device_type
import math

# define globals
SKIP_LINES = 2
COSCLEN = 51		# length of cosc filter
EPS = 1.0e-5	# termination threshold
#cos = np.array([0.f,-0.00020937301404,-0.00083238644375,-0.00187445134867,-0.003352219513758,-0.005284158713234,-0.007680040381756,-0.010530536243981,-0.013798126870435,-0.017410416484704,-0.021256733995966,-0.025188599234624,-0.029024272810166,-0.032557220569071,-0.035567944643756,-0.037838297355557,-0.039167132882787,-0.039385989227318,-0.038373445436298,-0.036066871845685,-0.032470479106137,-0.027658859359265,-0.02177557557417,-0.015026761314847,-0.007670107630023,0.0,0.007670107630023,0.015026761314847,0.02177557557417,0.027658859359265,0.032470479106137,0.036066871845685,0.038373445436298,0.039385989227318,0.039167132882787,0.037838297355557,0.035567944643756,0.032557220569071,0.029024272810166,0.025188599234624,0.021256733995966,0.017410416484704,0.013798126870435,0.010530536243981,0.007680040381756,0.005284158713234,0.003352219513758,0.00187445134867,0.00083238644375,0.00020937301404,0.0]).astype(float)
g_heir = []	
record_stress = False

def loadCSV(filename):
	data = np.array([])
	num_of_points = 0
	n_original_dims = 0
	with open(filename, "r") as reader:
		header = reader.readline()
		tokens = header.split(",")
		n_original_dims = len(tokens)

		reader.readline()

		for line in reader:
			tokens = np.array(line.split(",")).astype(float)
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



def mds(fileName, test_size, record_stress):
	myfile = open(fileName, "r")
	near_set_size = 4
	random_set_size = 4

	n_projection_dims = 2
	highD, num_of_points, n_original_dims = loadCSV(fileName)
	print(np.shape(highD))

	group_size = 128

	if(test_size):
		num_of_points = test_size

	num_of_groups = math.ceil(num_of_points / float(group_size))
	velocity = np.zeros(num_of_points * n_projection_dims)  
	force = np.zeros(num_of_points * n_projection_dims)
	seed_memory = np.random.randint(2000, size = num_of_points)
	pivot_indices = np.random.randint(num_of_points, size = num_of_points * (near_set_size + random_set_size))
	hd_distances = 1.3 * np.ones(num_of_points * (near_set_size + random_set_size))
	ld_distances = 1.3 * np.ones(num_of_points * (near_set_size + random_set_size))
	lowD = np.ones(num_of_points * n_projection_dims) * (float(np.random.randint(0, 32767)%10000)/10000.0-0.5)
	metadata = np.zeros(48)
	metadata[30] = 12345.0;
	metadata[31] = 12345.0;
	metadata[32] = 12345.0;
	resultN = np.zeros(num_of_groups)
	resultD = np.zeros(num_of_groups)

	metadata[0] = 0 

	print("data size: " + str(num_of_points) + "X" + str(n_original_dims))
	abc = 0	

	spring_force = .7 
	damping = .3
	delta_time = 0.3 
	freeness = .85
	size_factor = 1.0 / (float (near_set_size + random_set_size))

	err = 0
	#srand(2016);


	normalize(highD, num_of_points, n_original_dims)
	shuffle(highD, num_of_points, n_original_dims)



	high = 0.0

	correct = 0               # number of correct results returned


	platforms = cl.get_platforms()
	devices = platforms[0].get_devices(cl.device_type.GPU)
	context = cl.create_some_context()
	commands = cl.CommandQueue(context)
	program = cl.Program(context, open("/home/sajal/gitspace/Claret/source/device/claret_kernel.cl").read()).build()
	force_kernel = program.compute_force
	ld_kernel = program.compute_lowdist
	stress_kernel = program.computeStress

	print(force_kernel)




mds("/home/sajal/gitspace/Claret/data/breast-cancer-wisconsin.csv", 0, 0)




