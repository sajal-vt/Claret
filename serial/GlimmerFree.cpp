// GlimmerFree.cpp : Defines the entry point for the console application.
//

#include "Globals.h"
#include "Communicator.h"

// glimmer.cpp : Console program to compute Glimmer CPU MDS on a set of input coordinates
//				
//				Stephen Ingram (sfingram@cs.ubc.ca) 02/08
//

// cosc filter 
float cosc[] = {0.f,  -0.00020937301404f,      -0.00083238644375f,      -0.00187445134867f,      -0.003352219513758f,     -0.005284158713234f,     -0.007680040381756f,     -0.010530536243981f,     -0.013798126870435f,     -0.017410416484704f,     -0.021256733995966f,     -0.025188599234624f,     -0.029024272810166f,     -0.032557220569071f,     -0.035567944643756f,     -0.037838297355557f,     -0.039167132882787f,     -0.039385989227318f,     -0.038373445436298f,     -0.036066871845685f,     -0.032470479106137f,     -0.027658859359265f,     -0.02177557557417f,      -0.015026761314847f,     -0.007670107630023f,     0.f,      0.007670107630023f,      0.015026761314847f,      0.02177557557417f,       0.027658859359265f,      0.032470479106137f,      0.036066871845685f,      0.038373445436298f,      0.039385989227318f,      0.039167132882787f,      0.037838297355557f,      0.035567944643756f,      0.032557220569071f,      0.029024272810166f,      0.025188599234624f,      0.021256733995966f,      0.017410416484704f,      0.013798126870435f,      0.010530536243981f,      0.007680040381756f,      0.005284158713234f,      0.003352219513758f,      0.00187445134867f,       0.00083238644375f,       0.00020937301404f,       0.f};
float sstress[MAX_ITERATION];	// sparse stress calculation

/*
32 bit random number generation (default is 16 bit)
*/
int myrand( ) {

	unsigned int n = (unsigned int)rand();
	unsigned int m = (unsigned int)rand();

	return ((int)((n << 16) + m));
}


/*
distance and index comparison functions for qsort
*/
int distcomp( const void *a, const void *b ) {

	const INDEXTYPE *da = (const INDEXTYPE *)a;
	const INDEXTYPE *db = (const INDEXTYPE *)b;
	if(da->highd == db->highd)
		return 0;
	return (da->highd - db->highd)<0.f?-1:1;
}
int idxcomp( const void *a, const void *b ) {

	const INDEXTYPE *da = (const INDEXTYPE *)a;
	const INDEXTYPE *db = (const INDEXTYPE *)b;
	return (int)(da->index - db->index);
}

float max( float a, float b) {

	return (a < b)?b:a;
}
float min( float a, float b) {

	return (a < b)?a:b;
}


/*
Sparse Stress Termination Condition
*/
int terminate( INDEXTYPE *idx_set, int size ) {

	float numer = 0.f; // sq diff of dists
	float denom = 0.f; // sq dists
	float temp  = 0.f;

	if( iteration > MAX_ITERATION ) {

		return 1;
	}

	// compute sparse stress
	for( int i = 0; i < size; i++ ) {

		for( int j = 0; j < (V_SET_SIZE+S_SET_SIZE); j++ ) {

			temp	= (idx_set[i*(V_SET_SIZE+S_SET_SIZE) + j].highd==1.000f)?0.f:(idx_set[i*(V_SET_SIZE+S_SET_SIZE) + j].highd - idx_set[i*(V_SET_SIZE+S_SET_SIZE) + j].lowd);
			numer	+= temp*temp;
			denom	+= (idx_set[i*(V_SET_SIZE+S_SET_SIZE) + j].highd==1.000f)?0.f:(idx_set[i*(V_SET_SIZE+S_SET_SIZE) + j].highd * idx_set[i*(V_SET_SIZE+S_SET_SIZE) + j].highd);
		}
	}
	sstress[ iteration ] = numer / denom;
	//std::cout <<  sstress[iteration] << endl;

	// convolve the signal
	float signal = 0.f;
	if( iteration - stop_iteration > COSCLEN ) {

		for( int i = 0; i < COSCLEN; i++ ) {

			signal += sstress[ (iteration - COSCLEN)+i ] * cosc[ i ];
		}

		if( fabs( signal ) < EPS ) {

			stop_iteration = iteration;
			return 1;
		}
	}

	return 0;
}

/*
calculate the cosine distance between two points in g_vec_data
*/
float calc_cos_dist( int p1, int p2 ) {

	float dot = 0.f;

	for( int i = 0; i < g_vec_dims; i++ ) {
		for( int j = 0; j < g_vec_dims; j++ ) {

			if( g_vec_data[p1*g_vec_dims+i].index == g_vec_data[p2*g_vec_dims+j].index ) {

				dot += g_vec_data[p1*g_vec_dims+i].value * g_vec_data[p2*g_vec_dims+j].value;
			}
		}
	}

	return (1.f - dot)*(1.f - dot);
}

/*
Compute Chalmers' an iteration of force directed simulation on subset of size 'size' holding fixedsize fixed
*/
void force_directed( int size, int fixedsize ) {

	// initialize index sets
	if( iteration == stop_iteration ) {
		//std::cout << "Stop iteration " << iteration << " and " << stop_iteration << endl;
		for( int i = 0; i < size; i++ ) {

			for( int j = 0; j < V_SET_SIZE; j++ ) {

				g_idx[i*(V_SET_SIZE+S_SET_SIZE) + j ].index = myrand()%(g_interpolating?fixedsize:size);
			}
		}
	}

	// perform the force simulation iteration
	float *dir_vec		= (float *)malloc( sizeof(float) * n_embedding_dims );
	float *relvel_vec	= (float *)malloc( sizeof(float) * n_embedding_dims );
	float diff			= 0.f;
	float norm			= 0.f;
	float lo			= 0.f;
	float hi			= 0.f;

	// compute new forces for each point
	//std::cout << "still working " << endl;
	for( int i = fixedsize; i < size; i++ ) {

		for( int j = 0; j < V_SET_SIZE+S_SET_SIZE; j++ ) {

			// update the S set with random entries
			if( j >= V_SET_SIZE ) {

				g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].index = myrand()%(g_interpolating?fixedsize:size);
			}

			// calculate high dimensional distances
			int idx = g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].index;
			if( g_vec_dims ) {

				g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].highd = calc_cos_dist( idx, i );
			}
			else {

				hi = 0.f;
				for( int k = 0; k < n_original_dims; k++ ) {

					norm = (g_data[idx*n_original_dims+k] - g_data[i*n_original_dims+k]);
					hi += norm*norm;
				}
				g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].highd=(float)sqrt(hi);
			}
		}

		// sort index set by index
		qsort(&(g_idx[i*(V_SET_SIZE+S_SET_SIZE)]), (V_SET_SIZE+S_SET_SIZE), sizeof(INDEXTYPE), idxcomp );

		// mark duplicates (with 1000)
		for( int j = 1; j < V_SET_SIZE+S_SET_SIZE; j++ ) {

			if( g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].index==g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j-1].index )
				g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].highd=1000.f;
		}

		// sort index set by distance
		qsort(&(g_idx[i*(V_SET_SIZE+S_SET_SIZE)]), (V_SET_SIZE+S_SET_SIZE), sizeof(INDEXTYPE), distcomp );

		// move the point
		for( int j = 0; j < (V_SET_SIZE+S_SET_SIZE); j++ ) {

			// get a reference to the other point in the index set
			int idx = g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].index;

			norm = 0.f;
			for( int k = 0; k < n_embedding_dims; k++ ) {

				// calculate the direction vector
				dir_vec[k] =  g_embed[idx*n_embedding_dims+k] - g_embed[i*n_embedding_dims+k];
				norm += dir_vec[k]*dir_vec[k];
			}
			norm = sqrt( norm );
			g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].lowd = norm;
			if( norm > 1.e-6 && g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].highd!=1000.f ) {		// check for zero norm or mark

				// normalize direction vector
				for( int k = 0; k < n_embedding_dims; k++ ) {

					dir_vec[k] /= norm;
				}

				// calculate relative velocity
				for( int k = 0; k < n_embedding_dims; k++ ) {
					relvel_vec[k] = g_vel[idx*n_embedding_dims+k] - g_vel[i*n_embedding_dims+k];
				}

				// calculate difference between lo and hi distances
				lo = g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].lowd;	
				hi = g_idx[i*(V_SET_SIZE+S_SET_SIZE)+j].highd;
				diff = (lo - hi) * SPRINGFORCE;					

				// compute damping value
				norm = 0.f;
				for( int k = 0; k < n_embedding_dims; k++ ) {

					norm += dir_vec[k]*relvel_vec[k];
				}
				diff += norm*DAMPING;

				// accumulate the force
				for( int k = 0; k < n_embedding_dims; k++ ) {

					g_force[i*n_embedding_dims+k] += dir_vec[k]*diff;
				}
			}
		}

		// scale the force by the size factor
		for( int k = 0; k < n_embedding_dims; k++ ) {

			g_force[i*n_embedding_dims+k] *= SIZE_FACTOR;
		}
	}

	//std::cout << "computing velocities " << endl;

	// compute new velocities for each point with Euler integration
	for( int i = fixedsize; i < size; i++ ) {

		for( int k = 0; k < n_embedding_dims; k++ ) {

			float foo = g_vel[i*n_embedding_dims+k];
			float bar = foo + g_force[i*n_embedding_dims+k]*DELTATIME;
			float baz = bar * FREENESS;
			g_vel[i*n_embedding_dims+k] = max( min(baz, 2.0 ), -2.0 );
		}
	}

	// compute new positions for each point with Euler integration
	for( int i = fixedsize; i < size; i++ ) {
		for( int k = 0; k < n_embedding_dims; k++ ) {

			g_embed[i*n_embedding_dims+k] += g_vel[i*n_embedding_dims+k]*DELTATIME;
		}
	}

	// clean up memory allocation
	free(dir_vec);
	free(relvel_vec);
}


/*
init embedding to a random initialization in (-1,1) x (-1,1)
*/
void init_embedding( float *embedding ) {

	for( int i = 0; i < N; i++ ) {
		for( int j = 0; j < n_embedding_dims; j++ ) {
			embedding[i*(n_embedding_dims)+j]=((float)(rand()%10000)/10000.f)-0.5f;
		}
	}
}


/*
computes the input level heirarchy and size
*/
int fill_level_count( int input, int *h ) {

	static int levels = 0;
	//printf("h[%d]=%d\n",levels,input);
	h[levels]=input;
	levels++;
	if( input <= MIN_SET_SIZE )
		return levels;
	return fill_level_count( input / DEC_FACTOR, h );
}

/*
main function
*/
int main(/*int argc, char* argv[]*/)
	//int _tmain(int argc, _TCHAR* argv[])
{
	Communicator comm; 
	int argc = 5;
	char* argv[5];
	//argc = 3;
	argv[0] = "glimmer";
	argv[1] = "G:\\Coding\\glimmer\\data\\breast-cancer-wisconsin.csv";
	argv[2] = "G:\\Coding\\glimmer\\data\\breast-cancer-wisconsin_out.csv";
	argv[3] = "csv";
	argv[4] = "chalm";
	// check command line arguments
	if( argc < MIN_NUM_ARGS ) {
		std::printf("usage:  %s <inputfile> <outputfile> <type>", argv[0]);
		exit( 0 );
	}

	float *data = NULL;
	if( !strcmp( argv[3], "csv" ) ) {

		// load input CSV file
		
		data = comm.loadCSV((char *)argv[1]);
		g_data = data;
		int dum;
	}

	// begin timing -------------------------------------BEGIN TIMING
	clock_t start_time1 = clock();

	// allocate embedding and associated data structures
	g_levels = fill_level_count( N, g_heir );
	g_current_level = g_levels-1;
	g_embed	= (float *)malloc(sizeof(float)*n_embedding_dims*N);
	g_vel	= (float *)calloc(n_embedding_dims*N,sizeof(float));
	g_force	= (float *)calloc(n_embedding_dims*N,sizeof(float));
	g_idx	= (INDEXTYPE *)malloc(sizeof(INDEXTYPE)*N*(V_SET_SIZE+S_SET_SIZE));

	// initialize embedding
	init_embedding( g_embed );

	if( argc > 4 ) {
		std::cout << "if chosen" << endl;
		if( !strcmp( argv[4], "chalm" ) ) {
			g_chalmers = 1;
			for( int i = 0; i < N; i++ ) {
				std::cout << "current iteration " << i << endl;
				force_directed( N, 0 );
			}
		}
	} 

	clock_t start_time2 = clock();

	std::printf("%d %d", N, (start_time2-start_time1));

	if (strcmp(argv[2],"NONE")) {
		comm.outputCSV(argv[2],g_embed);
	}

	cout << endl;

	int dum;
	std::cin >> dum;
	// quit
	return 0;
}




