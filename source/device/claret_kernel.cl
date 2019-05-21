unsigned int rand(int* seed);
void swap (__private float* a,__private float* b );
int partition (__private float array[],__private float array1[], int l, int h);
void quickSortIterative (__private float array[], __private float array1[], int l, int h);

__kernel void compute_weighted_force(
  __global float* highD, 
  __global float* weights, // n_original_dims dimensional
  __global float* lowD_a,
    __global float* lowD_b,
  __global float* velocity_a,
    __global float* velocity_b,
  __global float* force, 
  __global int* seed_memory,
  __global unsigned int* pivot_indices, 
  const int start_index,
  const int end_index,
  const int n_original_dims,
  const int n_projection_dims,
  const int near_set_size,
  const int random_set_size,
    __global float* hd_distances,
  __global float* metadata,
    const int iteration
    ) 
{ 
  const float spring_force = 0.7f;
  const float damping = 0.3f;
  const float delta_time = 0.3f;
  const float freeness = 0.85f;
  __private float force_l[] = {0.f, 0.f};

  float size_factor = (1.f / ((float)(near_set_size + random_set_size)));
  int mod_op = start_index;
  if(start_index == 0)mod_op = end_index;
  
  int gid = get_global_id(0) + start_index;
  if(gid >= end_index)return;
  __private unsigned int my_pivot_indices[8];
  int pivot_size = near_set_size + random_set_size;

    int gid_pivot = gid * pivot_size;

  for(int i = 0; i < near_set_size; i++)
  {
    my_pivot_indices[i] = pivot_indices[gid_pivot + i];
  }
    
    

  __private float pivot_distances_high[8];
  __private float pivot_distances_low[8];
  __private float dir_vector[2];
  __private float rel_velocity[2];


    __private int seed = ((gid + iteration) * 123);
    __private rand_index = rand(&seed) % mod_op;

    my_pivot_indices[near_set_size] = rand_index;
    for(int j = 1; j <= 3; j++ )
    {
        my_pivot_indices[near_set_size + j] = (rand_index + j) % mod_op;             
    }
    

  for(int i = 0; i < pivot_size; i++)
  {
    __private float hi = 0.f;
    for( int k = 0; k < n_original_dims; k++ ) {

      __private float norm = highD[gid * n_original_dims + k] - highD[my_pivot_indices[i] * n_original_dims + k];
      hi += (weights[k] * norm * norm);
    }
    pivot_distances_high[i] = sqrt((float)hi);
  }

  quickSortIterative( my_pivot_indices, pivot_distances_high, 0, pivot_size - 1);


  // mark duplicates with 1000
  for(int i = 1; i < pivot_size; i++) {
    if(my_pivot_indices[i] == my_pivot_indices[i - 1]) {
      pivot_distances_high[i] = 1000.f;
    }
  }

  // sort pivot_distances and pivot_indices
  quickSortIterative( pivot_distances_high, my_pivot_indices, 0, pivot_size - 1);

  for(int i = 0; i < pivot_size; i++)
  {
    hd_distances[gid * pivot_size + i] = pivot_distances_high[i];
  }
  // Move the point
  __private int gid_proj_dims = gid * n_projection_dims;
  for(int i = 0; i < pivot_size; i++) {
    int idx = my_pivot_indices[i];
        int idx_proj_dims = idx * n_projection_dims;
  float norm = 0.f;
    for(int k = 0; k < n_projection_dims; k++) {
            if(iteration % 2 == 0) {
          dir_vector[k] = lowD_a[idx_proj_dims + k] - lowD_a[gid_proj_dims + k];
            } else{
                dir_vector[k] = lowD_b[idx_proj_dims + k] - lowD_b[gid_proj_dims + k];

            }
            
            norm += dir_vector[k] * dir_vector[k];
    }
      float norm_prev = norm; 
    norm = sqrt(norm);

    pivot_distances_low[i] = norm;

    if(norm > 1.e-6 && pivot_distances_high[i] != 1000.f)
    {
      for(int k = 0; k < n_projection_dims; k++) {
        dir_vector[k] /= norm;
      }
    
  
      // relative velocity
      for(int k = 0; k < n_projection_dims; k++) {
              if(iteration % 2 == 0) {
                    rel_velocity[k] = velocity_a[idx_proj_dims + k] - velocity_a[gid_proj_dims + k];
            
            } else {
                     rel_velocity[k] = velocity_b[idx_proj_dims + k] - velocity_b[gid_proj_dims + k]; 
                }

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
              force_l[k] += dir_vector[k] * delta_distance;
      }
    } 
  }

  // scale the force_l by size factor
  for(int k = 0; k < n_projection_dims; k++) {
    //force[gid_proj_dims + k] *= size_factor;
    force_l[k] *= size_factor;
  }

  for(int i = 0; i < pivot_size; i++)
  {
    pivot_indices[gid_pivot + i] = my_pivot_indices[i];
  }

   // update new velocity
    // v = v0 + at
    __private float v, v0;    
    for(int k = 0; k < n_projection_dims; k++) {
        
        if(iteration % 2 == 0) {
            v0 = velocity_a[gid_proj_dims + k];
        } else {
            v0 = velocity_b[gid_proj_dims + k];
        }

        v = v0 + force_l[k] * delta_time;
      // if(isnan(force_l[k]))metadata[30] = gid;
   //v = v0 + force[gid_proj_dims + k] * delta_time;
        v *= freeness;
        if(iteration % 2 == 0) {
            velocity_b[gid_proj_dims + k] = max(min(v, 2.f), -2.f);
       
        } else {
            velocity_a[gid_proj_dims + k] = max(min(v, 2.f), -2.f);
        }
    }
    
     // update new positions
     // x = x0 + vt
     for(int k = 0; k < n_projection_dims; k++) {
        if(iteration % 2 == 0) {
             lowD_b[gid_proj_dims + k] = lowD_a[gid_proj_dims + k] + velocity_b[gid_proj_dims + k] * delta_time;
        } else {
             lowD_a[gid_proj_dims + k] = lowD_b[gid_proj_dims + k] + velocity_a[gid_proj_dims + k] * delta_time;

             }
        
     }   
}


__kernel void compute_force(
	__global float* highD, 
	__global float* lowD_a,
    __global float* lowD_b,
	__global float* velocity_a,
    __global float* velocity_b,
	__global float* force, 
	__global int* seed_memory,
	__global unsigned int* pivot_indices, 
	const int start_index,
	const int end_index,
	const int n_original_dims,
	const int n_projection_dims,
	const int near_set_size,
	const int random_set_size,
    __global float* hd_distances,
	__global float* metadata,
    const int iteration
    ) 
{ 
	const float spring_force = 0.7f;
	const float damping = 0.3f;
	const float delta_time = 0.3f;
	const float freeness = 0.85f;
  __private float force_l[] = {0.f, 0.f};

	float size_factor = (1.f / ((float)(near_set_size + random_set_size)));
	int mod_op = start_index;
	if(start_index == 0)mod_op = end_index;
	
	int gid = get_global_id(0) + start_index;
	if(gid >= end_index)return;
	__private unsigned int my_pivot_indices[8];
	int pivot_size = near_set_size + random_set_size;
    //int seed = (unsigned int) seed_memory[gid];
    int gid_pivot = gid * pivot_size;
//    int seed = ((gid  * ( iteration + 1000) ) + 3000) % 2345;
	for(int i = 0; i < near_set_size; i++)
	{
		my_pivot_indices[i] = pivot_indices[gid_pivot + i];
	}
    
    

	__private float pivot_distances_high[8];
	__private float pivot_distances_low[8];
	__private float dir_vector[2];
	__private float rel_velocity[2];

    /* revert back
  for(int i = near_set_size; i < pivot_size; i++)
	{
		seed = seed + i;
		my_pivot_indices[i] = rand(&seed) % mod_op; 
	}
    */
  
    //__private rand_index = rand(&seed) % mod_op;
    __private int seed = ((gid + iteration) * 123);
    __private rand_index = rand(&seed) % mod_op;
    //__private rand_index = ((gid + iteration) * 123) % mod_op;
    my_pivot_indices[near_set_size] = rand_index;
    for(int j = 1; j <= 3; j++ )
    {
        my_pivot_indices[near_set_size + j] = (rand_index + j) % mod_op;             
    }
    
//	seed_memory[gid] = seed + iteration; 

	for(int i = 0; i < pivot_size; i++)
	{
		__private float hi = 0.f;
		for( int k = 0; k < n_original_dims; k++ ) {

			__private float norm = highD[gid * n_original_dims + k] - highD[my_pivot_indices[i] * n_original_dims + k];
			hi += norm * norm;
		}
		pivot_distances_high[i] = sqrt((float)hi);
	}

	quickSortIterative( my_pivot_indices, pivot_distances_high, 0, pivot_size - 1);


	// mark duplicates with 1000
	for(int i = 1; i < pivot_size; i++) {
		if(my_pivot_indices[i] == my_pivot_indices[i - 1]) {
			pivot_distances_high[i] = 1000.f;
		}
	}

	// sort pivot_distances and pivot_indices
	quickSortIterative( pivot_distances_high, my_pivot_indices, 0, pivot_size - 1);

	for(int i = 0; i < pivot_size; i++)
	{
		hd_distances[gid * pivot_size + i] = pivot_distances_high[i];
	}
	// Move the point
	__private int gid_proj_dims = gid * n_projection_dims;
	for(int i = 0; i < pivot_size; i++) {
		int idx = my_pivot_indices[i];
        int idx_proj_dims = idx * n_projection_dims;
	float norm = 0.f;
		for(int k = 0; k < n_projection_dims; k++) {
            if(iteration % 2 == 0) {
			    dir_vector[k] = lowD_a[idx_proj_dims + k] - lowD_a[gid_proj_dims + k];
           //     if(isnan(lowD_a[idx_proj_dims + k]) || isnan(lowD_a[gid_proj_dims + k])){metadata[33] = gid;}
            } else{
                dir_vector[k] = lowD_b[idx_proj_dims + k] - lowD_b[gid_proj_dims + k];
         //   if(isnan(lowD_b[idx_proj_dims + k]) || isnan(lowD_b[gid_proj_dims + k])){metadata[33] = gid;}
            }
            
            norm += dir_vector[k] * dir_vector[k];
		}
	    float norm_prev = norm;	
		norm = sqrt(norm);
       /* if(isnan(norm)){
            metadata[31] = gid;
            metadata[32] = norm_prev;
           }
           */
		pivot_distances_low[i] = norm;
        //if(gid == 100)metadata[i] = norm;
		if(norm > 1.e-6 && pivot_distances_high[i] != 1000.f)
		{
			for(int k = 0; k < n_projection_dims; k++) {
				dir_vector[k] /= norm;
  //              if(isnan(dir_vector[k])) {
    //                //metadata[30] = gid;
      //              //metadata[31] = dir_vector[k];
        //        }
			}
		
	
			// relative velocity
			for(int k = 0; k < n_projection_dims; k++) {
	            if(iteration % 2 == 0) {
                    rel_velocity[k] = velocity_a[idx_proj_dims + k] - velocity_a[gid_proj_dims + k];
            
		        } else {
                     rel_velocity[k] = velocity_b[idx_proj_dims + k] - velocity_b[gid_proj_dims + k]; 
                }
    /*            if(isnan(rel_velocity[k])){
                    metadata[30] = -1971.f;
                    metadata[35] = idx_proj_dims + k;
                    metadata[36] = gid_proj_dims + k;
                    metadata[37] = velocity_a[idx_proj_dims + k] - velocity_a[gid_proj_dims + k];
  
                }
*/
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
	            //force[gid_proj_dims + k] += dir_vector[k] * delta_distance;
	            force_l[k] += dir_vector[k] * delta_distance;
			}
		}	
	}

	// scale the force_l by size factor
	for(int k = 0; k < n_projection_dims; k++) {
		//force[gid_proj_dims + k] *= size_factor;
		force_l[k] *= size_factor;
	}

	for(int i = 0; i < pivot_size; i++)
	{
		pivot_indices[gid_pivot + i] = my_pivot_indices[i];
	}

   // update new velocity
    // v = v0 + at
    __private float v, v0;    
    for(int k = 0; k < n_projection_dims; k++) {
        
        if(iteration % 2 == 0) {
            v0 = velocity_a[gid_proj_dims + k];
        } else {
            v0 = velocity_b[gid_proj_dims + k];
        }

        v = v0 + force_l[k] * delta_time;
      // if(isnan(force_l[k]))metadata[30] = gid;
   //v = v0 + force[gid_proj_dims + k] * delta_time;
        v *= freeness;
        if(iteration % 2 == 0) {
            velocity_b[gid_proj_dims + k] = max(min(v, 2.f), -2.f);
       
        } else {
            velocity_a[gid_proj_dims + k] = max(min(v, 2.f), -2.f);
        }
    }
    
     // update new positions
     // x = x0 + vt
     for(int k = 0; k < n_projection_dims; k++) {
        if(iteration % 2 == 0) {
             lowD_b[gid_proj_dims + k] = lowD_a[gid_proj_dims + k] + velocity_b[gid_proj_dims + k] * delta_time;
  //           if(isnan(velocity_b[gid_proj_dims + k]))metadata[32] = gid;
        } else {
             lowD_a[gid_proj_dims + k] = lowD_b[gid_proj_dims + k] + velocity_a[gid_proj_dims + k] * delta_time;
//                if(isnan(lowD_b[gid_proj_dims +k]))metadata[32] = gid;
             }
        
     }   
}

__kernel void compute_lowdist(
                        __global float* lowD_a,
                        __global float* lowD_b,
						__global unsigned int * pivot_indices,
						__global float* ld_distances,
						const int start_index,
						const int end_index,
                        const int n_projection_dims,
						const int near_set_size,
						const int random_set_size,
                        const int iteration
)
{    
    int gid = get_global_id(0) + start_index;
    if(gid >= end_index)return;
	int pivot_size = near_set_size + random_set_size;
	float dir_vector[2];
    int gid_proj_dims = gid * n_projection_dims;
    int gid_pivot = gid * pivot_size;
	// compute ld_distances
	for(int i = 0; i < pivot_size; i++)
	{
		unsigned int idx = pivot_indices[gid * pivot_size + i];
        int idx_proj_dims = idx * n_projection_dims;
        float norm = 0.f;
		for(int k = 0; k < n_projection_dims; k++) {
            if(iteration % 2 == 0) {
                dir_vector[k] = lowD_b[idx_proj_dims + k] - lowD_b[gid_proj_dims + k];
            } else {
                dir_vector[k] = lowD_a[idx_proj_dims + k] - lowD_a[gid_proj_dims + k];
            }
			norm += dir_vector[k] * dir_vector[k];
		}
		ld_distances[gid_pivot + i] = sqrt(norm);
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
    float hd = hd_distances[global_index] == 1000.f ? 0.f : hd_distances[global_index];
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


unsigned int rand(int* seed) // 1 <= *seed < m
{
    unsigned long const a = 16807; //ie 7**5
    unsigned long const m = 2147483647; //ie 2**31-1
    unsigned int s = *seed;	
    unsigned long temp  = (s * a) % m;
    *seed = temp;
    return(*seed);
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
void quickSortIterative (__private float array[], __private float array1[], int l, int h)
{
    // Create an auxiliary stack
    int stack[8];

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

