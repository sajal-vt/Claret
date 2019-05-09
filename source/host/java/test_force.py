from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

import csv
import random
import numpy.random as nprod
import math
import sys

# simulation parameters
near_set_size = 4;
random_set_size = 4;
num_of_points = 1024;
n_original_dims = 10;
n_projection_dims = 2;

# simulation constants
spring_force = 0.7;
damping = 0.3;
delta_time = 0.3;
freeness = 0.85;
size_factor = 1.0 / ((float) (near_set_size + random_set_size));

platforms = cl.get_platforms()
print(platforms)


def loadSource( filePathName ):
        pfile = open(filePathName, 'r+')
        source = pfile.read()
        pfile.close()
        return source

def loadCSV( file_name, num_of_points, n_original_dims ):
        with open(file_name, 'rb') as csv_file:
                data_reader = csv.reader(csv_file)
                data_reader.next()
                data_reader.next()
                data = list(data_reader)

        return data

def generateRandomList( size ):
        data = [((item % 10000) / 10000) - 0.5 for item in random.sample(xrange(0, 10000), size)]
        return data

data = loadCSV("breast-cancer-wisconsin.csv", 5, 3)
print(len(data))
print(len(data[0]))

num_of_points = len(data)
n_original_dims = len(data[0])

group_size = 128.0
num_of_groups = (np.int32)(math.ceil(num_of_points / group_size))
print("num of groups", num_of_groups)

# declare data
HD = np.asarray([item for sublist in data for item in sublist])

highD = np.asarray([0.0] * len(HD)).astype(np.float32)
for i in xrange(0, len(highD)) :
        #print(i, float(highD[i]))
        if HD[i] == '?' :
                HD[i] = '0'
        highD[i] = float(HD[i])

#print(highD)
lowD = np.asarray(generateRandomList(num_of_points * n_projection_dims)).astype(np.float32)
seed_memory = np.asarray(random.sample(xrange(1, 2001), num_of_points)).astype(np.float32)
velocity = np.asarray([0.0] * (num_of_points * n_projection_dims)).astype(np.float32)
force = np.asarray([0.0] * (num_of_points * n_projection_dims)).astype(np.float32)
pivot_indices = np.asarray( [ np.random.randint(num_of_points, size = num_of_points * (near_set_size + random_set_size)) ]).astype(np.float32)
pivot_indices = pivot_indices[0]
hd_distances = np.asarray([1.0] * num_of_points * (near_set_size + random_set_size)).astype(np.float32)
ld_distances = np.asarray([1.0] * num_of_points * (near_set_size + random_set_size)).astype(np.float32)
meta_data = np.asarray([0.0] * 48).astype(np.float32)
resultN = np.asarray([0.0] * num_of_groups).astype(np.float32)
resultD = np.asarray([0.0] * num_of_groups).astype(np.float32)

print(lowD.shape)
print(highD.shape)
print(seed_memory.shape)
print(pivot_indices.shape)
print(hd_distances.shape)
print(ld_distances.shape)
#i initialize opencl environment
PYOPENCL_CTX='0'
platforms = cl.get_platforms()
print(platforms)

print(platforms[0].get_info(cl.platform_info.NAME))
devices = platforms[0].get_devices(device_type = cl.device_type.ALL)
print(devices)

#context = cl.create_some_context() #need to customize later
context = cl.Context(dev_type = cl.device_type.GPU,
properties = [(cl.context_properties.PLATFORM, platforms[0])] )
commands = cl.CommandQueue(context)
cl_source = loadSource("glimmer_kernel.cl")
program = cl.Program(context, cl_source).build()
force_kernel = program.relax_a_point
position_kernel = program.updatePosition
ld_kernel = program.computeLdDistance
stress_kernel = program.computeStress



mf = cl.mem_flags
d_highD = cl.Buffer(context, mf.READ_WRITE, size = highD.nbytes)
d_lowD = cl.Buffer(context, mf.READ_WRITE , size = lowD.nbytes)
d_velocity = cl.Buffer(context, mf.READ_WRITE, size = velocity.nbytes)
d_force = cl.Buffer(context, mf.READ_WRITE, size = force.nbytes)
d_seed_memory = cl.Buffer(context, mf.READ_WRITE, size = seed_memory.nbytes)
d_pivot_indices = cl.Buffer(context, mf.READ_WRITE, size = pivot_indices.nbytes)
d_hd_distances = cl.Buffer(context, mf.READ_WRITE, size = hd_distances.nbytes)
d_ld_distances = cl.Buffer(context, mf.READ_WRITE, size = ld_distances.nbytes)
d_meta_data = cl.Buffer(context, mf.READ_WRITE, size = meta_data.nbytes)
d_resultN = cl.Buffer(context, mf.READ_WRITE, size = resultN.nbytes)
d_resultD = cl.Buffer(context, mf.READ_WRITE, size = resultD.nbytes)


start_index = 0
end_index = num_of_points - 1

lws = [num_of_points, 1]

cl.enqueue_write_buffer(commands, d_highD, highD, is_blocking = True)
cl.enqueue_write_buffer(commands, d_lowD, lowD, is_blocking = True)
cl.enqueue_write_buffer(commands, d_velocity, velocity, is_blocking = True)
cl.enqueue_write_buffer(commands, d_force, force, is_blocking = True)
cl.enqueue_write_buffer(commands, d_seed_memory, seed_memory, is_blocking = True)
cl.enqueue_write_buffer(commands, d_pivot_indices, pivot_indices, is_blocking = True)
cl.enqueue_write_buffer(commands, d_hd_distances, hd_distances, is_blocking = True)
cl.enqueue_write_buffer(commands, d_ld_distances, ld_distances, is_blocking = True)
cl.enqueue_write_buffer(commands, d_meta_data, meta_data, is_blocking = True)
cl.enqueue_write_buffer(commands, d_resultN, resultN, is_blocking = True)
cl.enqueue_write_buffer(commands, d_resultD, resultD, is_blocking = True)


force_kernel(commands, lws, None, d_highD, d_lowD, d_velocity, d_force, d_seed_memory, d_pivot_indices, np.int32(start_index), np.int32(end_index), np.int32(n_original_dims), np.int32(n_projection_dims), np.int32(near_set_size), np.int32(random_set_size), d_hd_distances, d_meta_data)

print(hd_distances)
cl.enqueue_read_buffer(commands, d_hd_distances, hd_distances, is_blocking = True)
print(hd_distances)

def level_force_directed(
        highD,
        lowD,
        d_lowD,
        pivot_indices,
        hd_istances,
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
        position_kernel,
        ld_kernel,
        stress_kernel,
        resultN,
        resultD,
        d_resultN,
        d_resultD,
        num_of_groups
):
        group_size = 128.0
        local_work_size = [ (np.int32)(group_size), 0, 0 ]
        global_work_size = [ (np.int32)(math.ceil((end_index - start_index) / group_size) * group_size), 0, 0 ];
     
                print("HD before", hd_distances)
                force_kernel(
                commands, global_work_size, None, d_highD, d_lowD, d_velocity, d_force, d_seed_memory,
                d_pivot_indices, (np.int32)(start_index), (np.int32)(end_index), (np.int32)(n_original_dims),
                (np.int32)(n_projection_dims), (np.int32)(near_set_size), (np.int32)(random_set_size),
                d_hd_distances, d_meta_data)

                cl.enqueue_read_buffer(commands, d_hd_distances, hd_distances, is_blocking = True)
                print("HD after", hd_distances)



g_interpolating = False
level_force_directed(
                highD, lowD, d_lowD, pivot_indices, hd_distances, ld_distances,
                d_hd_distances, d_ld_distances, d_pivot_indices, num_of_points,
                n_original_dims, n_projection_dims, 0, num_of_points - 1, g_interpolating,  near_set_size,
                random_set_size, commands, force_kernel, position_kernel,
                ld_kernel, stress_kernel, resultN, resultD, d_resultN, d_resultD, num_of_groups)

-- INSERT --                                                                                                   