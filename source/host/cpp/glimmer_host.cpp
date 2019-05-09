#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <set>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

using namespace std;

char * loadSource(char *filePathName, size_t *fileSize);
void randomMemInit(float* data, int size);
void printArray(float* array, int start, int end, int dimension);
float computeStress(float* highD, float* lowD, int num_of_points, int n_original_dims, int n_projection_dims, int s, int e);
float distance(float* a, float* b, int dim);
float distance(int i, int j, float* highD, int dimension);
bool testPivots(int num_of_points, float* highD, float* prevPivot, float* pivot, float* output, int dimension);
float avgNNDistance(int pointIdx, float* highD, float* pivot, int dimension);
int testForce(int num_of_points, float* highD, float* lowD, float* velocity, float* pivot, float* prevForce, float* refForce, float* highDistance );
bool testLowDistance(int pointIdx, float* lowD, float* pivot, float* output);
int testPosition(int num_of_points, float* prevVelocity, float* velocity, float* prevLowD, float* lowD, float* force);
int countNewPivots(int pointIdx, std::set<float> &currentPivots, unsigned int* pivot_indices);
void normalize(float* data, int size, int dimension);
void shuffle(float* data, int size, int dimension);
unsigned int myrand( );
float* loadCSV( const char *filename, int& num_of_points, int& n_original_dims);
bool terminate(int iteration, int stop_iteration, float* sstress);

void level_force_directed(
	float* highD, 
	float* lowD,
	cl_mem& d_lowD_a,
    cl_mem& d_lowD_b,
	unsigned * pivot_indices,
	float* hd_distances,
	float* ld_distances,
	cl_mem& d_hd_distances,
	cl_mem& d_ld_distances,
	cl_mem& d_pivot_indices,
	int num_of_points,
	int n_original_dims,
	int n_projection_dims,
	int start_index,
	int end_index, 
	bool interpolate, 
	int near_set_size,
	int random_set_size,
	cl_command_queue& commands,
	cl_kernel& force_kernel,
	cl_kernel& ld_kernel,
	cl_kernel& stress_kernel,
	float* resultN,
	float* resultD,
	cl_mem& d_resultN,
	cl_mem& d_resultD,
	int num_of_groups,
    float* velocity,
    float* force,
    cl_mem& d_velocity_a,
    cl_mem& d_velocity_b,
    cl_mem& d_force,
    int& iteration,
    float* metadata,
    cl_mem& d_metadata);
	
int fill_level_count( int input, int *h );

#define SKIP_LINES 2
#define COSCLEN			51		// length of cosc filter
#define EPS				1.e-5f	// termination threshold
float cosc[] = {0.f,  -0.00020937301404f,      -0.00083238644375f,      -0.00187445134867f,      -0.003352219513758f,     -0.005284158713234f,     -0.007680040381756f,     -0.010530536243981f,     -0.013798126870435f,     -0.017410416484704f,     -0.021256733995966f,     -0.025188599234624f,     -0.029024272810166f,     -0.032557220569071f,     -0.035567944643756f,     -0.037838297355557f,     -0.039167132882787f,     -0.039385989227318f,     -0.038373445436298f,     -0.036066871845685f,     -0.032470479106137f,     -0.027658859359265f,     -0.02177557557417f,      -0.015026761314847f,     -0.007670107630023f,     0.f,      0.007670107630023f,      0.015026761314847f,      0.02177557557417f,       0.027658859359265f,      0.032470479106137f,      0.036066871845685f,      0.038373445436298f,      0.039385989227318f,      0.039167132882787f,      0.037838297355557f,      0.035567944643756f,      0.032557220569071f,      0.029024272810166f,      0.025188599234624f,      0.021256733995966f,      0.017410416484704f,      0.013798126870435f,      0.010530536243981f,      0.007680040381756f,      0.005284158713234f,      0.003352219513758f,      0.00187445134867f,       0.00083238644375f,       0.00020937301404f,       0.f};
int g_heir[50];	
bool record_stress;

int main(int argc, char** argv)
{
	ofstream myfile;
	myfile.open("shuttle-output.csv");	
	struct timeval start, end;

	int near_set_size = 4;
	int random_set_size = 4;
	int num_of_points = 1024; 
	int n_original_dims = 10;
	int n_projection_dims = 2;
    int select_file = atoi(argv[1]);
    int test_size = atoi(argv[2]);
    record_stress = atoi(argv[3]);   
    float* highD;
     switch(select_file) {
          case 0:
               highD = loadCSV("../data/breast-cancer-wisconsin.csv",
                                num_of_points,
                                n_original_dims);
               printf("-INFO- Using breast-cancer-wisconsin\n");
               break;
          case 1:
               highD = loadCSV("../data/shuttle_trn_corr.csv",
                                num_of_points,
                                n_original_dims);
               printf("-INFO- Using shutte-trn-corr\n");
               break;
           case 2:
               highD = loadCSV("../data/output20topics.csv",
                                num_of_points,
                                n_original_dims);
               printf("-INFO- Using 20 topics \n");
               break;
           case 3:
               highD = loadCSV("../data/output100topics.csv",
                                num_of_points,
                                n_original_dims);
               printf("-INFO- Using 100 topics \n");
               break;
           default:
               printf("-ERROR- PROVIDE CORRECT FILE\n");
               exit(1);
      }
//	float* highD = loadCSV("breast-cancer-wisconsin.csv", num_of_points, n_original_dims);
	
//	float* highD = loadCSV("shuttle_trn_corr.csv", num_of_points, n_original_dims);

	int group_size = 128;
    if(test_size)
    {
        num_of_points = test_size;
    }
	int num_of_groups = ceil(num_of_points / (float)group_size );
	
	float* velocity = new float[num_of_points * n_projection_dims];  
	float* force = new float[num_of_points * n_projection_dims];

	int* seed_memory = new int[num_of_points];
	unsigned int* pivot_indices = new unsigned int[num_of_points * (near_set_size + random_set_size)];
	float* hd_distances = new float[num_of_points * (near_set_size + random_set_size)]();
	float* ld_distances = new float[num_of_points * (near_set_size + random_set_size)]();
	float* lowD = new float[num_of_points * n_projection_dims];
	float* metadata = new float[48];
    metadata[30] = 12345.f;
    metadata[31] = 12345.f;
    metadata[32] = 12345.f;
	float* resultN = new float[num_of_groups]();
	float* resultD = new float[num_of_groups]();

	metadata[0] = 0;  
	
	std::cout << "data size: " << num_of_points << "X" << n_original_dims << std::endl;
	int abc;

	float spring_force = .7f; 
	float damping = .3f; 
	float delta_time = 0.3f; 
	float freeness = .85f;
	float size_factor = 1.f / ((float) (near_set_size + random_set_size));

	int err;  
	srand(2016);


    normalize(highD, num_of_points, n_original_dims);
	shuffle(highD, num_of_points, n_original_dims);
    std::cout << "here" << std::endl;



	float high = 0.f;

	for(int i = 0; i < num_of_points * n_projection_dims; i++)
	{
		lowD[i] = ((float)(rand()%10000)/10000.f)-0.5f;
//        std::cout << lowD[i] << std::endl;
	}

	for(int i = 0; i < num_of_points; i++)
	{
		seed_memory[i] = rand() % 2000;
	}


	for(int i = 0; i < num_of_points * n_projection_dims; i++)
	{
		velocity[i] = 0.f;
		force[i] = 0.f;
	}

	for(int i = 0;  i < num_of_points * (near_set_size + random_set_size); i++)
	{
		pivot_indices[i] = (rand() % num_of_points);
		hd_distances[i] = 1.3f;
		ld_distances[i] = 1.3f;
	}

	unsigned int correct;               // number of correct results returned

	size_t global;                      // global domain size for our calculation
	size_t local;                       // local domain size for our calculation

	cl_platform_id platform_ids[2];
	cl_device_id device_id;             // compute device id 
	cl_device_id devices[2];
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel force_kernel;                   // compute kernel
	cl_kernel ld_kernel;
	cl_kernel stress_kernel;


	cl_mem d_highD;
	cl_mem d_lowD_a;
    cl_mem d_lowD_b;
	cl_mem d_velocity_a;
    cl_mem d_velocity_b;
	cl_mem d_force;
	cl_mem d_seed_memory;
	cl_mem d_pivot_indices;
	cl_mem d_hd_distances;
	cl_mem d_ld_distances;
	cl_mem d_metadata;
	cl_mem d_resultN;
	cl_mem d_resultD;


	int gpu = 1;
	cl_uint numPlatforms;
	cl_int status;

    
	err = clGetPlatformIDs(2, platform_ids, &numPlatforms);
	std::cout << "number of platforms: " << numPlatforms << std::endl; 
	if (err != CL_SUCCESS)
	{
		printf("Error in clGetPlatformID, %d\n", err);
	}

	char buffer[10240];
	clGetPlatformInfo(platform_ids[0], CL_PLATFORM_NAME, 10240, buffer, NULL);
	std::cout << "Platform[1] Name: " << buffer << std::endl;

	clGetPlatformInfo(platform_ids[1], CL_PLATFORM_NAME, 10240, buffer, NULL);
    	std::cout << "Platform[2] Name: " << buffer << std::endl;
	
    std::cout << "Running forward MDS on GPU " << std::endl;

    cl_uint num_devices;

	err = clGetDeviceIDs(platform_ids[1] , CL_DEVICE_TYPE_ALL, 1, devices, &num_devices);
    std::cout << "num of devices: " << num_devices << std::endl;

    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 10240, buffer, NULL);
    std::cout << "first device name: " << buffer << std::endl;
    //clGetDeviceInfo(devices[1], CL_DEVICE_NAME, 10240, buffer, NULL);
    std::cout << "second device name: " << buffer << std::endl;


	err = clGetDeviceIDs(platform_ids[1] , CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source buffer
	size_t sourceFileSize;
	//char kernel_file[] = "glimmer_kernel.cl";
	char kernel_file[] = "../source/device/glimmer_kernel.cl";
    char *cSourceCL = loadSource(kernel_file, &sourceFileSize);
	program = clCreateProgramWithSource(context, 1, (const char **) & cSourceCL, &sourceFileSize, &err);

	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
    cl_int ret = clBuildProgram(program, 1, &device_id, "-D __OPENCLCC__ -I . -D FOO", NULL, NULL);
    if (ret != CL_SUCCESS) {fprintf(stderr, "Error in clBuildProgram: %d!\n", ret); ret = CL_SUCCESS; }
    size_t logsize = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
    char * log = (char *) malloc (sizeof(char) *(logsize+1));
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logsize, log, NULL);
    fprintf(stderr, "CL_PROGRAM_BUILD_LOG:\n%s", log);
    free(log);

	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	force_kernel = clCreateKernel(program, "compute_force", &err);
	ld_kernel = clCreateKernel(program, "compute_lowdist", &err);
	stress_kernel = clCreateKernel(program, "computeStress", &err);
	if (!force_kernel || !ld_kernel || !stress_kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	std::cout << "assigning device buffers " << std::endl;
	// Create the input and output arrays in device memory for our calculation

	d_highD =  clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * num_of_points * n_original_dims, NULL, &err);
	d_lowD_a = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_lowD_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_velocity_a = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_velocity_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_force = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * n_projection_dims, NULL, &err);
	d_seed_memory = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_of_points, NULL, &err);
	d_pivot_indices = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int) * num_of_points * (near_set_size + random_set_size), NULL, &err);
	d_hd_distances = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * (near_set_size + random_set_size), NULL, &err);
	d_ld_distances = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_points * (near_set_size + random_set_size), NULL, &err);
	d_metadata = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 48, NULL, &err);
	d_resultN = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_groups, NULL, &err);
	d_resultD = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_groups, NULL, &err);


	if (!d_highD || !d_lowD_a || !d_lowD_b || !d_velocity_a || !d_velocity_b || !d_force || !d_pivot_indices || !d_seed_memory || !d_hd_distances || !d_ld_distances || !d_metadata)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    


	//Launch OpenCL kernel
	size_t localWorkSize[3] = {32, 0, 0}, globalWorkSize[3] = {num_of_points, 0, 0};

	int iteration = 0;
	int start_index = 0;
	int end_index = num_of_points;
	
	err = clSetKernelArg(force_kernel, 0, sizeof(cl_mem), (void *)&d_highD);
	err |= clSetKernelArg(force_kernel, 1, sizeof(cl_mem), (void *)&d_lowD_a);
	err |= clSetKernelArg(force_kernel, 2, sizeof(cl_mem), (void *)&d_lowD_b);
    err |= clSetKernelArg(force_kernel, 3, sizeof(cl_mem), (void *)&d_velocity_a);
    err |= clSetKernelArg(force_kernel, 4, sizeof(cl_mem), (void *)&d_velocity_b);
	err |= clSetKernelArg(force_kernel, 5, sizeof(cl_mem), (void *)&d_force);
	err |= clSetKernelArg(force_kernel, 6, sizeof(cl_mem), (void *)&d_seed_memory);
	err |= clSetKernelArg(force_kernel, 7, sizeof(cl_mem), (void *)&d_pivot_indices);
	err |= clSetKernelArg(force_kernel, 8, sizeof(int), (void *)&start_index);
	err |= clSetKernelArg(force_kernel, 9, sizeof(int), (void *)&end_index);
	err |= clSetKernelArg(force_kernel, 10, sizeof(int), (void *)&n_original_dims);
	err |= clSetKernelArg(force_kernel, 11, sizeof(float), (void *)&n_projection_dims);
	err |= clSetKernelArg(force_kernel, 12, sizeof(float), (void *)&near_set_size);
	err |= clSetKernelArg(force_kernel, 13, sizeof(float), (void *)&random_set_size);
	err |= clSetKernelArg(force_kernel, 14, sizeof(cl_mem), (void*)&d_hd_distances);
	err |= clSetKernelArg(force_kernel, 15, sizeof(cl_mem), (void*)&d_metadata);
    
    err |= clSetKernelArg(ld_kernel, 0, sizeof(cl_mem), (void*)&d_lowD_a);
    err |= clSetKernelArg(ld_kernel, 1, sizeof(cl_mem), (void*)&d_lowD_b);
   	err |= clSetKernelArg(ld_kernel, 2, sizeof(cl_mem), (void*)&d_pivot_indices);
    err |= clSetKernelArg(ld_kernel, 3, sizeof(cl_mem), (void*)&d_ld_distances);
    err |= clSetKernelArg(ld_kernel, 4, sizeof(int), (void*)&start_index);
	err |= clSetKernelArg(ld_kernel, 5, sizeof(int), (void*)&end_index);
	err |= clSetKernelArg(ld_kernel, 6, sizeof(int), (void*)&n_projection_dims);
    err |= clSetKernelArg(ld_kernel, 7, sizeof(int), (void*)&near_set_size);
    err |= clSetKernelArg(ld_kernel, 8, sizeof(float), (void*)&random_set_size);


	std::cout << "Initial number of groups: " << num_of_groups << std::endl;


	err |= clSetKernelArg(stress_kernel, 0, sizeof(cl_mem), (void *)&d_hd_distances);
    err |= clSetKernelArg(stress_kernel, 1, sizeof(cl_mem), (void *)&d_ld_distances);               
    err |= clSetKernelArg(stress_kernel, 2, sizeof(cl_float) * group_size, NULL);    
    err |= clSetKernelArg(stress_kernel, 3, sizeof(cl_float) * group_size, NULL);
    err |= clSetKernelArg(stress_kernel, 4, sizeof(int), (void *)&num_of_points);
    err |= clSetKernelArg(stress_kernel, 5, sizeof(cl_mem), (void *)&d_resultN);
    err |= clSetKernelArg(stress_kernel, 6, sizeof(cl_mem), (void *)&d_resultD);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	err &= clEnqueueWriteBuffer(commands, d_pivot_indices, CL_TRUE, 0, sizeof(unsigned int) * num_of_points * (near_set_size + random_set_size), pivot_indices, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_highD, CL_TRUE, 0, sizeof(float) * num_of_points * n_original_dims, highD, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_lowD_a, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, lowD, 0, NULL, NULL);
    err &= clEnqueueWriteBuffer(commands, d_lowD_b, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, lowD, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_velocity_a, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, velocity, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_velocity_b, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, velocity, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_seed_memory, CL_TRUE, 0, sizeof(int) * num_of_points, seed_memory, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_metadata, CL_TRUE, 0, sizeof(float) * 48, metadata, 0, NULL, NULL);
	err &= clEnqueueWriteBuffer(commands, d_force, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, force, 0, NULL, NULL); 
	err &= clEnqueueWriteBuffer(commands, d_hd_distances, CL_TRUE, 0, sizeof(float) * num_of_points * (near_set_size + random_set_size), hd_distances, 0, NULL, NULL);

	//Multi-level code
	bool g_done = false;
	bool g_interpolating = false;
	int stop_iteration = 0;
	int g_levels = fill_level_count( num_of_points, g_heir );
	int g_current_level = g_levels-1;

    int iteration_pp = 0;
    long st = time(NULL);
    gettimeofday(&start, NULL);
	while( !g_done ) {
		if( g_interpolating ) // interpolate
		{
			cout << "Interpolating " << g_heir[g_current_level] << " and " << g_heir[g_current_level + 1] << endl;
			num_of_groups = ceil(g_heir[g_current_level] / (float)group_size);
			std::cout << "number of groups : " << num_of_groups << std::endl;
			level_force_directed(
                    highD, 
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
                    d_metadata);
	                  


		}
		else
		{
			cout << "Relaxing " << g_heir[g_current_level] << " and 0" << endl; 
			num_of_groups = ceil(g_heir[g_current_level] / (float)group_size);
			std::cout << "number of groups : " << num_of_groups << std::endl;	
			level_force_directed(
                    highD, 
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
                    d_metadata);
	    ;


		}
		if( true ) {

			if( g_interpolating ) {
				g_interpolating = false;
			}
			else {
				g_current_level--; // move to the next level down
				g_interpolating = true;

                if( g_current_level < 0 ) {
					cout << "done " << endl;
					g_done = true;
				}
			}
		}

		iteration++;	// increment the current iteration count			
	}
	gettimeofday(&end, NULL);
	err &= clEnqueueReadBuffer(commands, d_lowD_a, CL_TRUE, 0, sizeof(float) * num_of_points * n_projection_dims, lowD, 0, NULL, NULL);
	long et = time(NULL);
	std::cout << "elapsed time: " << (et - st) << std::endl;
	//Multi-level code
	double elapsed = (end.tv_sec - start.tv_sec) * 1000 +
              ((end.tv_usec - start.tv_usec)/1000.0);
        std::cout << "elapsed time: " << elapsed << " ms " << std::endl;

	for(int i = 0; i < num_of_points; i++)
	{
		myfile << lowD[i * n_projection_dims + 0] << "," << lowD[i * n_projection_dims + 1] << "\n";
	}
	myfile.close();

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	printf("Matrix multiplication completed...\n"); 

	//Shutdown and cleanup
	delete[] highD;
	delete[] lowD;
	delete[] velocity;
	delete[] force;
	delete[] seed_memory;
	delete[] pivot_indices;
	delete[] hd_distances;
	delete[] ld_distances;
	delete[] metadata;
	
	clReleaseMemObject(d_highD);
	clReleaseMemObject(d_lowD_a);
    clReleaseMemObject(d_lowD_b);
	clReleaseMemObject(d_velocity_a);
    clReleaseMemObject(d_velocity_b);
	clReleaseMemObject(d_force);
	clReleaseMemObject(d_seed_memory);
	clReleaseMemObject(d_pivot_indices);
	clReleaseMemObject(d_hd_distances);
	clReleaseMemObject(d_ld_distances);
	clReleaseMemObject(d_metadata);
	clReleaseProgram(program);
	clReleaseKernel(force_kernel);
	clReleaseKernel(ld_kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}

char * loadSource(char *filePathName, size_t *fileSize)
{
	FILE *pfile;
	size_t tmpFileSize;
	char *fileBuffer;
	pfile = fopen(filePathName, "rb");

	if (pfile == NULL)
	{
		printf("Open file %s open error!\n", filePathName);
		return NULL;
	}

	fseek(pfile, 0, SEEK_END);
	tmpFileSize = ftell(pfile);

	fileBuffer = (char *)malloc(tmpFileSize);

	fseek(pfile, 0, SEEK_SET);
	fread(fileBuffer, sizeof(char), tmpFileSize, pfile);

	fclose(pfile);
	*fileSize = tmpFileSize;
	return fileBuffer;
}

// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
	int i;

	for (i = 0; i < size; ++i)
		data[i] = rand() % 10; // / RAND_MAX;
}


void printArray(float* array, int start, int end, int dimension)
{
	for(int i = start; i < end; i++)
	{
		for(int d = 0; d < dimension; d++)
		{
			if(array[i * dimension + d] >= 0)continue;
			std::cout << array[i * dimension + d] << ", ";
		}
    }
}

float computeStress(float* highD, float* lowD, int num_of_points, int n_original_dims, int n_projection_dims, int s, int e) {
	float stress = 0.f;
	for(int i = s; i < e; i++)
	{
		for(int j = i + 1; j < e; j++)
		{
			//float hd = distance(highD + i * n_original_dims, highD + j * n_original_dims, n_original_dims);
			//float ld = distance(lowD + i * n_projection_dims, lowD + j * n_projection_dims, n_projection_dims);
			float hd = distance(i, j, highD, n_original_dims);
			float ld = distance(i, j, lowD, n_projection_dims);
			float delta =  fabs(hd - ld);
			stress += delta;
		}
	}

	return stress;
}

float computeSparseStress(float* highD, float* lowD, unsigned int* pivot_indices, int num_of_points, int near_set_size, int random_set_size, int n_original_dims, int n_projection_dims, int s, int e) {
	float numerator = 0.f;
	float denominator = 0.f;
	int pivot_size = near_set_size + random_set_size;
	for(int i = 0; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			float hd = distance(i, pivot_indices[i * pivot_size + j], highD, n_original_dims);
			float ld = distance(i, pivot_indices[i * pivot_size + j], lowD, n_projection_dims);
			float delta =  (hd - ld);
			numerator += (delta * delta);
			denominator += hd * hd;
		}
	}
	return numerator / denominator;
}

float computeSparseStress(float* highD, float* lowD, float *hd_distances, float* ld_distances, float* pivot_indices, int num_of_points, int near_set_size, int random_set_size, int n_original_dims, int n_projection_dims, int s, int e) {
	float numerator = 0.f;
	float denominator = 0.f;
	int pivot_size = near_set_size + random_set_size;
	for(int i = s; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			//float hd = distance(i, (int)pivot_indices[i * pivot_size + j], highD, n_original_dims);
			
			//float hd = distance(i, (int)pivot_indices[i * pivot_size + j], highD, n_original_dims);
			float hd = hd_distances[i * pivot_size + j];
			float ld = ld_distances[i * pivot_size + j];
				
				
			//float ld = distance(i, (int)pivot_indices[i * pivot_size + j], lowD, n_projection_dims);
			float delta =  (hd - ld);
			numerator += (delta * delta);
			denominator += hd * hd;
		}
	}
	return numerator / denominator;
}

float computeAlternateStress(float* hd_distances, float* ld_distances, float* pivot_indices, int num_of_points, int near_set_size, int random_set_size, int n_original_dims, int n_projection_dims, int s, int e) {
	float numerator = 0.f;
	float denominator = 0.f;
	int pivot_size = near_set_size + random_set_size;
	
	for(int i = 0; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			float hd = hd_distances[i * pivot_size + j];
			if(hd == 1000.f)hd = 0.f;
			float ld = ld_distances[i * pivot_size + j];
			float delta =  (hd - ld);
			numerator += (delta * delta);
			denominator += hd * hd;
		}
	}
	return numerator / denominator;
}

float distance(float* a, float* b, int dim)
{
	float dist = 0.f;
	for(int d = 0; d < dim; d++)
	{
		dist += (a[d] - b[d]) * (a[d] - b[d]);
	}
	return sqrt(dist);
}

bool testLowDistance(int pointIdx, float* lowD, float* pivot, float* output)
{
	for(int i = 0; i < 24; i++)
	{
		float dist  = distance(pointIdx, (int)pivot[pointIdx * 24 + i], lowD, 2);
		std::cout << dist << " <> " << output[pointIdx * 24 + i] << std::endl;
	}
}

bool testPivots(int num_of_points, float* highD, float* prevPivot, float* pivot, float* output, int dimension)
{
bool sorted = true;
bool preserveNear = true;
//num_of_points = 1;
for(int pointIdx = 0; pointIdx < num_of_points; pointIdx++){
	float distances[24];
	float prev_dist = 0.f;

//	bool sorted = true;
	for(int i = 0; i < 24; i++)
	{
		distances[i] = distance(pointIdx, (int)pivot[pointIdx * 24 + i], highD, dimension);
//		std::cout << pivot[pointIdx * 24 + i] << ":" <<  distances[i] << " <>  " << output[pointIdx * 24 + i] << std::endl;
		if(prev_dist > distances[i])
		{
			std::cout << "reversal " << pointIdx << ":" << prev_dist << " <> " << distances[i]  << std::endl;
//			sorted = false;
			return false;
		}
		prev_dist = distances[i];
	}

	if(!sorted)std::cout << "Not sorted " << std::endl;
	preserveNear = false;
	int j = 0;
	int count = 0;
	for(int i = 0; i < 14; i++)
	{
		for(j = 0; j < 24; j++)
		{
			if(prevPivot[pointIdx * 24 + i] == pivot[pointIdx * 24 + j])count++;
		}
	}

	if(count >= 14)preserveNear = true;
	if(!preserveNear){ std::cout << pointIdx << ":near set not preserved " << count << std::endl; return false;} //std::cout << "Near set is not preserved " << std::endl;
}
	return sorted && preserveNear;
}


// highD is array of size 1024 x dimension
float distance(int i, int j, float* data, int dimension)
{
	//std::cout << "Distance between " << i << " and " << j << ": ";
	float norm = 0.f;
	for(int d = 0; d < dimension; d++)
	{
		float diff = (data[i * dimension + d] - data[j * dimension + d]);
		norm += diff * diff;
	}

	//std::cout << norm << std::endl;
	return (float)sqrt(norm);
}

void normalize(float* data, int size, int dimension)
{
	std::cout << "IN " << size << "--" << dimension << std::endl;
    float* max_vals = new float[dimension];
    float* min_vals = new float[dimension]; 
    for( int i = 0; i < dimension; i++ ) {
        max_vals[ i ] = 0.f;
        min_vals[ i ] = 10000.0f;
    }

	int dum;
//	cin >> dum;
    int k = 0;
    for( int i = 0; i < size; i++ ) {        
        for( int j = 0; j < dimension; j++ ) {
            if( data[i * (dimension) + j] > max_vals[j] ) {
                max_vals[j] = data[i * (dimension) +j];
            }
            if( data[i*(dimension)+j] < min_vals[j] ) {
                min_vals[j] = data[i*(dimension)+j];                    
            }
        }
    }
//	cin >> dum;
    for( int i = 0; i < dimension; i++ ) {
        max_vals[ i ] -= min_vals[ i ];
    }

    for( int i = 0; i < size; i++ ) {        
        for( int j = 0; j < dimension; j++ ) {
            if( (max_vals[j] - min_vals[j]) < 0.0001f ) {
                data[i*(dimension)+j] = 0.f;
            }
            else {
                data[i*(dimension)+j] = 
                    (data[i*(dimension)+j] - min_vals[j])/max_vals[j];
                if(  data[i*(dimension)+j] >= 1000.f || data[i * dimension + j] <= -1000  ) 
                    data[i*(dimension)+j] = 0.f;
            }
        }
    }
//cin >> dum;
    delete max_vals ;
    delete min_vals;
	std::cout << "OUT" << std::endl;
}

void shuffle(float* data, int size, int dimension)
{
    float *shuffle_temp = new float[dimension];
    int shuffle_idx = 0;
    for( int i = 0; i < size * dimension; i += dimension ) {

        shuffle_idx = i + ( myrand() % (size - (i / dimension)) ) * dimension;
        for( int j = 0; j < dimension; j++ ) {    // swap

            shuffle_temp[j]=data[i+j];
            data[i+j] = data[shuffle_idx+j];
            data[shuffle_idx+j] = shuffle_temp[j];
        }        
    }
    delete shuffle_temp;
}

unsigned int myrand( ) {

    unsigned int n = (unsigned int)rand();
    unsigned int m = (unsigned int)rand();
//	std::cout << "n and m: " << n << ", " << m << ":" << (int)((n << 16) + m) << std::endl;
//	return 5;
    unsigned int rv = ((unsigned int)((n << 16) + m));
    return rv;
    //std::cout << rv << std::endl;
    //return ((int)((n << 16) + m));
}


float* loadCSV( const char *filename, int& num_of_points, int& n_original_dims ) {

    char line[65536];    // line of input buffer
    char item[512];        // single number string
    float *data = NULL;    // output data

    // open the file 
    ifstream fp;
    fp.open( filename);


    // get dataset statistics
    int N = 0;
    n_original_dims = 0;

    while( fp.getline( line, 65535) != NULL && N < 43502) {

        // count the number of points (for every line)
        N++;

        // count the number of dimensions (once)
        if( n_original_dims == 0 && N > SKIP_LINES) {
            int i = 0;
            while( line[i] != '\0' ) {
                if( line[i] == ',' ) {
                    n_original_dims++;
                }
                i++;
            }
            n_original_dims++;
        }
    }
    fp.close();
    std::cout << "number of data points " << N << " and " << n_original_dims;
    N -= SKIP_LINES;

    // allocate our data buffer    
    data = (float*)malloc(sizeof(float)*N*n_original_dims);

    // read the data into the buffer
    fp.open(filename);
    int skip = 0;
    int k = 0;
	int c = 0;
    while( fp.getline( line, 65535) != NULL && c < 43502 ) {

        int done = 0;
        int i = 0;
        int j = 0;
        while( !done ) {

            // skip the introductory lines
            if( skip++ < SKIP_LINES ) {

                done = 1;
            }
            else {

                // parse character data
                if( line[i] == ',' ) {

                    item[j] = '\0';
                    data[k++] = (float) atof( item );
                    j = 0;
                }
                else if( line[i] == '\n' || line[i] == '\0' ) {

                    item[j] = '\0';
                    data[k++] = (float) atof( item );
                    done++;
                }
                else if( line[i] != ' ' ) {

                    item[j++] = line[i];
                }
                i++;
            }
        }
	c++;
    }
    num_of_points = N;
    return data;
}

int verify_hd_distances(float* highD, unsigned int* pivot_indices, float* hd_distances, 
int s, int e, int n_original_dims, int near_set_size, int random_set_size)
{
	int count = 0;
	int pivot_size = near_set_size + random_set_size;
	for(int i = s; i < e; i++)
	{
		for(int j = 0; j < pivot_size; j++)
		{
			float hd = distance(i, (int)pivot_indices[i * pivot_size + j], highD, n_original_dims);
			if(hd != hd_distances[i * pivot_size + j])
			{
				count++;
				std::cout << hd_distances[i * pivot_size + j] << std::endl;
			}
		}
	}
	return count;
}

void level_force_directed(
	float* highD, 
	float* lowD,
	cl_mem& d_lowD_a,
    cl_mem& d_lowD_b,
	unsigned int* pivot_indices,
	float* hd_distances,
	float* ld_distances,
	cl_mem& d_hd_distances,
	cl_mem& d_ld_distances,
	cl_mem& d_pivot_indices,
	int num_of_points,
	int n_original_dims,
	int n_projection_dims,
	int start_index,
	int end_index, 
	bool interpolate, 
	int near_set_size,
	int random_set_size,
	cl_command_queue& commands,
	cl_kernel& force_kernel,
	cl_kernel& ld_kernel,
	cl_kernel& stress_kernel,
	float* resultN,
	float* resultD,
	cl_mem& d_resultN,
	cl_mem& d_resultD,
	int num_of_groups,
    float* velocity,
    float* force,
    cl_mem& d_velocity_a,
    cl_mem& d_velocity_b,
    cl_mem& d_force,
    int& iteration_pp,
    float* metadata,
    cl_mem& d_metadata)
{
	ofstream fout;
	if(record_stress)fout.open("stress.csv", std::ofstream::out | std::ofstream::app);	
	// Initialize near sets using random values
	int modular_operand = interpolate ? start_index : end_index;
	for(int i = 0; i < end_index; i++)
	{       
		for(int j = 0; j < near_set_size; j++)
		{       
			pivot_indices[i * (near_set_size + random_set_size) + j] = floor(rand() % modular_operand);
		}
	}
	
 //   int iteration_pp = 0;
	int group_size = 64;
	size_t localWorkSize[3] = {group_size, 0, 0};
	size_t globalWorkSize[3] = {(int)(ceil((end_index - start_index) / group_size) * group_size), 0, 0};

	int err = 0;

    err |= clSetKernelArg(force_kernel, 8, sizeof(int), (void *)&start_index);
    err |= clSetKernelArg(force_kernel, 9, sizeof(int), (void *)&end_index);
   	
    err |= clSetKernelArg(ld_kernel, 4, sizeof(int), (void*)&start_index);
    err |= clSetKernelArg(ld_kernel, 5, sizeof(int), (void*)&end_index);
	
	err |= clEnqueueWriteBuffer(commands, d_pivot_indices, CL_TRUE, 0, 
            sizeof(unsigned int) * num_of_points * (near_set_size + random_set_size), pivot_indices, 0, NULL, NULL);
	float* sstress = new float[end_index - start_index];

	int length = end_index * 8;
	err |= clSetKernelArg(stress_kernel, 4, sizeof(int), (void *)&length);
	

	size_t stress_g_size[3] = {(int)(ceil((length / 8.f) /(float) group_size) * group_size), 0, 0};
/*
    std::cout << "Before loop: 100th Points 2D Coordinate: " << lowD[100 * 2] << ", " << lowD[100 * 2 + 1] << std::endl;
	for(int i = 0; i < 8; i++) {
        std::cout << pivot_indices[800 + i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < 8; i++) {
        std::cout << hd_distances[800 +i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
*/
    // if(iteration_pp % 2 == 0)
/*            err &= clEnqueueReadBuffer(commands, d_velocity_b, CL_TRUE, 0, sizeof(float) * num_of_points * 2, velocity, 0, NULL, NULL);      
            for(int i = 0; i < num_of_points; i++){
                std::cout << i << ": " << velocity[i * 2 ] << ", " << velocity[i * 2 + 1] << std::endl;
            }    
            //    else
            err &= clEnqueueReadBuffer(commands, d_velocity_a, CL_TRUE, 0, sizeof(float) * num_of_points * 2, velocity, 0, NULL, NULL);
       for(int i = 0; i < num_of_points; i++){
                std::cout << i << ": " << velocity[i * 2 ] << ", " << velocity[i * 2 + 1] << std::endl;
            }    
  */          
for(int iteration = start_index; iteration < end_index; iteration++)
	{
		err |= clSetKernelArg(force_kernel, 16, sizeof(int), (void *)&iteration_pp);
        err |= clSetKernelArg(ld_kernel, 9, sizeof(int), (void *)&iteration_pp);

        err = clEnqueueNDRangeKernel(commands, force_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		err = clEnqueueNDRangeKernel(commands, ld_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		err = clEnqueueNDRangeKernel(commands, stress_kernel, 1, NULL, stress_g_size, localWorkSize, 0, NULL, NULL);	

	    err &= clEnqueueReadBuffer(commands, d_resultN, CL_TRUE, 0, sizeof(float) * num_of_groups, resultN, 0, NULL, NULL);
		err &= clEnqueueReadBuffer(commands, d_resultD, CL_TRUE, 0, sizeof(float) * num_of_groups, resultD, 0, NULL, NULL);
/* 	    err &= clEnqueueReadBuffer(commands, d_pivot_indices, CL_TRUE, 0, sizeof(unsigned int) * num_of_points * 8, pivot_indices, 0, NULL, NULL);
	    err &= clEnqueueReadBuffer(commands, d_hd_distances, CL_TRUE, 0, sizeof(float) * num_of_points * 8, hd_distances, 0, NULL, NULL);
        err &= clEnqueueReadBuffer(commands, d_metadata, CL_TRUE, 0, sizeof(float) * 48, metadata, 0, NULL, NULL);
        
        std::cout << "Inspecting metadata: " << metadata[30] << ", " << metadata[35] << ", " << metadata[36] << ", " << metadata[37] << std::endl;
        if(iteration_pp % 2 == 0)
            err &= clEnqueueReadBuffer(commands, d_lowD_b, CL_TRUE, 0, sizeof(float) * num_of_points * 2, lowD, 0, NULL, NULL);      
        else
            err &= clEnqueueReadBuffer(commands, d_lowD_a, CL_TRUE, 0, sizeof(float) * num_of_points * 2, lowD, 0, NULL, NULL);
      

        for(int i = 0; i < num_of_points; i++)
        {
        //    std::cout << i << ": " << lowD[i * 2] << ", " <<  lowD[ i * 2 + 1] << std::endl;
        }
        //std::cout << "lowd coordinates " << std::endl;
        for(int i = 0; i < 8; i++)
        {
            int idx = pivot_indices[800 + i];
        //    std::cout << lowD[idx * 2] << ", " << lowD[idx * 2 + 1] << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Computing distance between 100th and 443rd points in lowd " << pivot_indices[800] << std::endl;
        float sum = 0.f;
        for(int i = 0; i < n_projection_dims; i++)
        {
         //   std::cout << lowD[100 * n_projection_dims + i] << " - " << lowD[pivot_indices[800] * 2 + i] <<std::endl;
            float diff = lowD[100 * n_projection_dims + i] - lowD[pivot_indices[800] * 2 + i] ;
         //   std::cout << diff << std::endl;
         //   std::cout << diff * diff << std::endl;
            sum += (diff*diff);
        }

        std::cout << sum << std::endl;
        std::cout << sqrt(sum) << std::endl;
        for(int i = 0; i < 8; i++)
        {
        
            float cpu_distance = distance(100, pivot_indices[800 + i], lowD, n_projection_dims);
            //std::cout << metadata[i] << " <>  " << cpu_distance << std::endl;
        }
        std::cout << std::endl << std::endl;
        float cpu_stress = computeSparseStress(highD, lowD, pivot_indices, num_of_points, near_set_size, random_set_size, n_original_dims, n_projection_dims, start_index, end_index);
	

        /*
	    err &= clEnqueueReadBuffer(commands, d_pivot_indices, CL_TRUE, 0, sizeof(unsigned int) * num_of_points * 8, pivot_indices, 0, NULL, NULL);
		err &= clEnqueueReadBuffer(commands, d_hd_distances, CL_TRUE, 0, sizeof(float) * num_of_points * 8, hd_distances, 0, NULL, NULL);
    	err &= clEnqueueReadBuffer(commands, d_ld_distances, CL_TRUE, 0, sizeof(float) * num_of_points * 8, ld_distances, 0, NULL, NULL);

        if(iteration_pp % 2 == 0)
            err &= clEnqueueReadBuffer(commands, d_lowD_b, CL_TRUE, 0, sizeof(float) * num_of_points * 2, lowD, 0, NULL, NULL);      
        else
            err &= clEnqueueReadBuffer(commands, d_lowD_a, CL_TRUE, 0, sizeof(float) * num_of_points * 2, lowD, 0, NULL, NULL);
      
        std::cout << std::endl << "Debugging Iteration[" << iteration_pp << "]" << std::endl;

        for(int i = 0; i < 8; i++) {
        //    std::cout << pivot_indices[800 + i] << " ";
        }
        std::cout << "HD:" << std::endl;
        for(int i = 0; i < 8; i++) {
            float cpu_distance = distance(100, pivot_indices[800 + i], highD, n_original_dims);
            float gpu_distance = hd_distances[800 + i];
            std::cout << cpu_distance << "<>" << gpu_distance << std::endl;
        }
        
        std::cout << "HD:" << std::endl;
        for(int i = 0; i < 8; i++) {
            float cpu_distance = distance(100, pivot_indices[800 + i], lowD, n_projection_dims);
            float gpu_distance = ld_distances[800 + i];
            std::cout << cpu_distance << "<>" << gpu_distance << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;

/*
        err &= clEnqueueReadBuffer(commands, d_lowD_a, CL_TRUE, 0, sizeof(float) * num_of_points * 2, lowD, 0, NULL, NULL);
    //    std::cout << iteration_pp <<  ">> (a) In loop: 100th Points 2D Coordinate: " << lowD[100 * 2] << ", " << lowD[100 * 2 + 1] << std::endl;
        float x01, x02, v1, v2, x1, x2;
        
        if(iteration_pp % 2 == 0) {x01 = lowD[200]; x02 = lowD[201];}
        else {x1 = lowD[200]; x2 = lowD[201];}
       
        err &= clEnqueueReadBuffer(commands, d_lowD_b, CL_TRUE, 0, sizeof(float) * num_of_points * 2, lowD, 0, NULL, NULL);
    //    std::cout << iteration_pp <<  ">> (b) In loop: 100th Points 2D Coordinate: " << lowD[100 * 2] << ", " << lowD[100 * 2 + 1] << std::endl;
        
        if(iteration_pp % 2 == 0) {x1 = lowD[200]; x2 = lowD[201];}
        else {x01 = lowD[200]; x02 = lowD[201];}
        
        err &= clEnqueueReadBuffer(commands, d_velocity_a, CL_TRUE, 0, sizeof(float) * num_of_points * 2, velocity, 0, NULL, NULL);
      //  std::cout << iteration_pp <<  ">> (a) In loop: 100th Point's velocity: " << velocity[100 * 2] << ", " << velocity[100 * 2 + 1] << std::endl;
        if(iteration_pp % 2 == 1) {v1 = velocity[200]; v2 = velocity[201];}
       

        err &= clEnqueueReadBuffer(commands, d_velocity_b, CL_TRUE, 0, sizeof(float) * num_of_points * 2, velocity, 0, NULL, NULL);
        //std::cout << iteration_pp <<  ">> (b) In loop: 100th Point's velocity: " << velocity[100 * 2] << ", " << velocity[100 * 2 + 1] << std::endl;
        if(iteration_pp % 2 == 0) {v1 = velocity[200]; v2 = velocity[201];}

        err &= clEnqueueReadBuffer(commands, d_force, CL_TRUE, 0, sizeof(float) * num_of_points * 2, force, 0, NULL, NULL);
        //std::cout << iteration_pp <<  ">> (b) In loop: 100th Point's force: " << force[100 * 2] << ", " << force[100 * 2 + 1] << std::endl;

        std::cout << x1 << " <> " << x01+ v1 * 0.3f << "||" 
                << x2 << " <> " << x02 + v2 * 0.3f << std::endl;
       */
        iteration_pp++;
        if(err == CL_SUCCESS)
		{
			float d = 0.f;
			float n = 0.f;
			for(int k = 0; k < num_of_groups; k++)
			{
				n += resultN[k];
				d += resultD[k];
			} 
			float otherStress = sqrt(n / d);
	        //cout << "cpu <> gpu stress " << sqrt(cpu_stress) << " <> " << otherStress << std::endl;
            if(record_stress) {
                fout << otherStress << "\n";
            }
			sstress[iteration - start_index] = otherStress;
			if(terminate(iteration, start_index, sstress ))
			{
				std::cout << "Stopping at iteration with stress : " << iteration - start_index << ", " << otherStress << std::endl;
				break;
			}
		}
	}
	if(record_stress) {
        fout.close();
    }
	delete[] sstress;
}

bool terminate(int iteration, int stop_iteration, float* sstress)
{
	if(iteration - stop_iteration >= 400) return true;
	float signal = 0.f;
	if( iteration - stop_iteration > COSCLEN ) {

		for( int i = 0; i < COSCLEN; i++ ) {

			signal += sstress[ (iteration - COSCLEN)+i ] * cosc[ i ];
		}

		if( fabs( signal ) < EPS ) {
			return true;
		}
	}

	return false;
}

int fill_level_count( int input, int *h ) {
	static int levels = 0;
	printf("h[%d]=%d\n",levels,input);
	h[levels]=input;
	levels++;
	if( input <= 1000 )
		return levels;
	return fill_level_count( input / 8, h );
}
