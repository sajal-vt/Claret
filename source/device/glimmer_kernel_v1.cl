#define MAX_SET_SIZE    8
#define NEAR_SET_SIZE   MAX_SET_SIZE/2
#define RANDOM_SET_SIZE NEAR_SET_SIZE
#define FLT_MIN_TOLERANCE 1.0e-6f


/* Generates a random number based on a seed value */
unsigned int rand(unsigned int* seed);

/* Pointer swap */
void swap (__private float* a,
           __private float* b );

/*Helper function to quick-sort */
int partition (__private float array[],
               __private float array1[], 
               int l,  //Length
               int h); //Height

/* Quick Sort */
void quickSortIterative (__private float array[],
                         __private float array1[],
                         int l, 
                         int h);

__kernel void relax_a_point(
	__global float* highD,                  //High dimension data of NxD  
	__global float* lowD,                   //2D projection 
	__global float* velocity,               //Velocity is N*2
	__global float* force,                  //Force on each point Nx2
	__global int*   seed_memory,            //Generation of Random Set 
	__global unsigned int* pivot_indices,   //Eight for each point Nx8 
	const int       start_index,            //Start point in the array
	const int       end_index,              //End point in array 0->(N-1)
	const int       n_original_dims,        //Dimensions in original data
	const int       n_projection_dims,      //Dimensions in projected data (2)
	const int       near_set_size,          //Size of pivot indicies for near-data set
	const int       random_set_size,        //Random Set Size
    const int       iter_variable,          //Iteration count number
    __global float* hd_distances,           //Distance in high dimnesion
	__global float* metadata)               //NOT-USED, For debugging purposes
{ 
    //Below constants are selected based on earlier work by Chalmar
	const float spring_force = 0.7f; 
	const float damping      = 0.3f;
	const float delta_time   = 0.3f;
	const float freeness     = 0.85f;
	float size_factor  = (1.f / ((float)(near_set_size + random_set_size)));
	int mod_op;
    int gid;
    int pivot_size;
    unsigned int seed;
	__private unsigned int my_pivot_indices [MAX_SET_SIZE];     //Indices of 
	__private float pivot_distances_high    [MAX_SET_SIZE];
	__private float pivot_distances_low     [MAX_SET_SIZE];
	__private float dir_vector[2];
	__private float rel_velocity[2];

	mod_op     = start_index;
	pivot_size = near_set_size + random_set_size;
	gid        = get_global_id(0) + start_index;

	if(start_index == 0)
        mod_op = end_index;
	
	if(gid >= end_index)
        return;

	seed = (unsigned int) seed_memory[gid];
	
	for(int i = 0; i < near_set_size; i++)
	{
		my_pivot_indices[i] = pivot_indices[gid * pivot_size + i];
	}

    /*
    seed = my_pivot_indices[0] *
           my_pivot_indices[1] +
           my_pivot_indices[2] *
           my_pivot_indices[3];
    */

	for(int i = near_set_size; i < pivot_size; i++)
	{
		seed = seed + i;
		my_pivot_indices[i] = rand(&seed) % mod_op; 
	}

	seed_memory[gid] = seed; 


	for(int i = 0; i < pivot_size; i++)
	{
        __private float hi;
		hi = 0.f;
		for( int k = 0; k < n_original_dims; k++ ) {
         __private float norm;

		    norm = (highD[gid * n_original_dims + k] 
                   -highD[my_pivot_indices[i]*n_original_dims + k]);
			hi += norm * norm;
		}
		pivot_distances_high[i] = sqrt((float)hi);
	}

	quickSortIterative( my_pivot_indices, pivot_distances_high, 0, pivot_size - 1);


    //Remove the duplicates by assigning a ridculously large value
	for(int i = 1; i < pivot_size; i++) {
		if(my_pivot_indices[i] == my_pivot_indices[i - 1]) {
			pivot_distances_high[i] = FLT_MAX_10_EXP;
		}
	}

    //int replacement = 4;
    //for(int i = 0; i < near_set_size; i++) {
    //    if(pivot_distances_high[i] == FLT_MAX_10_EXP) {
    //        swap(&pivot_distances_high[i], &pivot_distances_high[replacement]);
    //        swap(& my_pivot_indices[i], & my_pivot_indices[replacement]);
    //        replacement++;
    //    }        
    //}
	// TODO: sort pivot_distances and pivot_indices
	quickSortIterative(pivot_distances_high, my_pivot_indices, 0, pivot_size - 1);

	for(int i = 0; i < pivot_size; i++)
	{
		hd_distances[gid * pivot_size + i] = pivot_distances_high[i];
	}

	// Move the point
	for(int i = 0; i < pivot_size; i++) {
		int idx = my_pivot_indices[i];
		float norm = 0.f;
		for(int k = 0; k < n_projection_dims; k++) {
			dir_vector[k] = lowD[idx * n_projection_dims + k] - lowD[gid * n_projection_dims + k];
			norm += dir_vector[k] * dir_vector[k];
		}
		
		norm = sqrt(norm);
		pivot_distances_low[i] = norm;

		if(norm > FLT_MIN_TOLERANCE && pivot_distances_high[i] != FLT_MAX_10_EXP)
		{
			for(int k = 0; k < n_projection_dims; k++) {
				dir_vector[k] /= norm;
			}
		
	
			// relative velocity
			for(int k = 0; k < n_projection_dims; k++) {
				rel_velocity[k] = velocity[idx * n_projection_dims + k] - velocity[gid * n_projection_dims + k];
			}

			// calculate difference
			float delta_distance = (pivot_distances_low[i] - pivot_distances_high[i]) * spring_force;
			// compute damping value
			norm = 0.f;
			for(int k = 0; k < n_projection_dims; k++) {
				norm += dir_vector[k] * rel_velocity[k];
			}
			delta_distance += norm * damping;
			
			// accumulate the force
			for(int k = 0; k < n_projection_dims; k++) {
	            		force[gid * n_projection_dims + k] += dir_vector[k] * delta_distance;
			}
		}	
	}

	// scale the force by size factor
	for(int k = 0; k < n_projection_dims; k++) {
		force[gid * n_projection_dims + k] *= size_factor;
	}

	for(int i = 0; i < pivot_size; i++)
	{
		pivot_indices[gid * pivot_size + i] = my_pivot_indices[i];
	}
}

__kernel void updatePosition(
                        __global float* velocity,
                        __global float* lowD,
                        __global float* force,
						const int       start_index,
						const int       end_index,
                        const int       n_projection_dims,
                        const float     delta_time,
                        const float     freeness)
{
    
    int gid = get_global_id(0) + start_index;
    float v0;
    float v;

    if(gid >= end_index)
        return;

    // update new velocity
    // v = v0 + at
    for(int k = 0; k < n_projection_dims; k++) {
        v0 = velocity[gid * n_projection_dims + k];
        v  = v0 + force[gid * n_projection_dims + k] * delta_time;
        v *= freeness;
        velocity[gid * n_projection_dims + k] = max(min(v, 2.f), -2.f);
    }
    
     // update new positions
     // x = x0 + vt
     for(int k = 0; k < n_projection_dims; k++) {
         lowD[gid * n_projection_dims + k] += velocity[gid * n_projection_dims + k] * 
                                              delta_time;
     }
     //barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void computeLdDistance(
                        __global float* lowD,
						__global float* pivot_indices,
						__global float* ld_distances,
						const int start_index,
						const int end_index,
                        const int n_projection_dims,
						const int near_set_size,
						const int random_set_size
)
{    
    int gid = get_global_id(0) + start_index;
    if(gid >= end_index)return;
	int pivot_size = near_set_size + random_set_size;
	float dir_vector[2];
	// compute ld_distances
	for(int i = 0; i < pivot_size; i++)
	{
		int idx = (int)pivot_indices[gid * pivot_size + i];
		float norm = 0.f;
		for(int k = 0; k < n_projection_dims; k++) {
			dir_vector[k] = lowD[idx * n_projection_dims + k] - lowD[gid * n_projection_dims + k];
			norm += dir_vector[k] * dir_vector[k];
		}
		ld_distances[gid * pivot_size + i] = sqrt(norm);
	}
}

__kernel
void computeStress(__global float* hd_distances,
		           __global float* ld_distances,		  
                   __local float* scratchN,
	               __local float* scratchD,
                   __const int length,
                   __global float* resultN,
	               __global float* resultD) {

  int global_index = get_global_id(0);
  float numerator = 0.f;
  float denominator = 0.f;
  
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float hd = hd_distances[global_index] == FLT_MAX_10_EXP ? 0.f : hd_distances[global_index];
    float ld = ld_distances[global_index];
	if(isinf(hd) || isnan(ld))hd = 0.f;
	if(isinf(ld) || isnan(ld))ld = 0.f;
    float tempN = hd - ld;
    numerator += (tempN * tempN);
    //float tempD = hd_distances[global_index];
    denominator += (hd * hd);
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratchN[local_index] = numerator;
  scratchD[local_index] = denominator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float mineN = scratchN[local_index];
      float mineD = scratchD[local_index];
      float otherN = scratchN[local_index + offset];
      float otherD = scratchD[local_index + offset];
if(isinf(mineN) || isnan(mineN))mineN = 0.f;
if(isinf(mineD) || isnan(mineD))mineD = 0.f;
if(isinf(otherN) || isnan(otherN))otherN = 0.f;
if(isinf(otherD) || isnan(otherD))otherD = 0.f;

      scratchN[local_index] = mineN + otherN;
      scratchD[local_index] = mineD + otherD;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
	if(isinf(scratchN[0]) || isnan(scratchN[0]))scratchN[0] = 0.f;
if(isinf(scratchD[0]) || isnan(scratchD[0]))scratchD[0] = 0.f;
    resultN[get_group_id(0)] = scratchN[0];
    resultD[get_group_id(0)] = scratchD[0];
  }
}


unsigned int rand(unsigned int* seed) // 1 <= *seed < m
{
    unsigned long const a = 16807; //ie 7**5
    unsigned long const m = 2147483647; //ie 2**31-1
    unsigned int        s = *seed;	
    unsigned long       temp  = (s * a) % m;
    *seed = temp;
    return(*seed);

    // unsigned int lfsr;
    // lfsr = *seed;

    // lfsr = ((((lfsr >> 31)   /*Shift the 32nd bit to the first bit*/
    //          ^(lfsr >> 6)    /*XOR it with the seventh bit*/
    //          ^(lfsr >> 4)    /*XOR it with the fifth bit*/
    //          ^(lfsr >> 2)    /*XOR it with the third bit*/
    //          ^(lfsr >> 1)    /*XOR it with the second bit*/
    //          ^lfsr)          /*and XOR it with the first bit.*/
    //           & 0x0000001)   /*Strip all the other bits off and*/
    //           <<31)          /*move it back to the 32nd bit.*/
    //         | (lfsr >> 1);   /*Or with the lfsr shifted right.*/
    // return lfsr;
}

// A utility function to swap two elements
void swap ( __private float* a, __private float* b )
{
    float t = *a;
    *a = *b;
    *b = t;
}


/* This function is same in both iterative and recursive*/
int partition (__private float array[], __private float array1[], int l, int h)
{
    float x = array[h];
    int i = (l - 1);

    for (int j = l; j <= h- 1; j++)
    {
        if (array[j] <= x)
        {
            i++;
            swap (&array[i], &array[j]);
            swap (&array1[i], &array1[j]);
        }
    }
    swap (&array[i + 1], &array[h]);
    swap (&array1[i + 1], &array1[h]);
    return (i + 1);
}




/* A[] --> Array to be sorted,
   l  --> Starting index,
   h  --> Ending index */
void quickSortIterative (__private float array[],
                         __private float array1[], 
                         int h, 
                         int l)
{
    // Create an auxiliary stack
    int stack[MAX_SET_SIZE];

    // initialize top of stack
    int top = -1;

    // push initial values of l and h to stack
    stack[ ++top ] = l;
    stack[ ++top ] = h;

    // Keep popping from stack while is not empty
    while ( top >= 0 )
    {
        // Pop h and l
        h = stack[ top-- ];
        l = stack[ top-- ];

        // Set pivot element at its correct position
        // in sorted array
        int p = partition( array, array1, l, h );

        // If there are elements on left side of pivot,
        // then push left side to stack
        if ( p-1 > l )
        {
            stack[ ++top ] = l;
            stack[ ++top ] = p - 1;
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if ( p+1 < h )
        {
            stack[ ++top ] = p + 1;
            stack[ ++top ] = h;
        }
    }
}

