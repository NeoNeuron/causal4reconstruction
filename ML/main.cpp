
/* Lorenz model: Network (Library and Regular)*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------


#include "Def.h"
#include "Random.h"
#include "Read_parameters.h"
#include "Initialization.h"
#include "Find_cubic_hermite_root.h"
#include "Runge_Kutta4.h"
#include "Run_model.h"
#include "Largest_Lyapunov.h"
#include "Delete.h"





int main(int argc, char **argv)
{
	long seed, seed0, seed1, seed2;
	clock_t t0, t1;
	char str[200];
	double MLE;
	double mean_fire_rate;


	Read_parameters(seed, seed1);
	if (argc == 5)
	{
		P_c = atof(argv[1]);
		S[0] = atof(argv[2]), S[1] = S[0], S[2] = S[0], S[3] = S[0];
		f[0] = atof(argv[3]); f[1] = f[0];
		Nu = atof(argv[4]);
	}
	else if (argc == 6)
	{
		P_c = atof(argv[1]);
		S[0] = atof(argv[2]); S[1] = S[0];
		S[2] = atof(argv[3]); S[3] = S[2];
		f[0] = atof(argv[4]); f[1] = f[0];
		Nu = atof(argv[5]);
	}

	//I_CONST = atoi(argv[5]);///
	//I_const_input = atof(argv[6]);



	/////////////////////	
	out_put_filename();
	seed0 = seed;    // Create connect matrix
	seed2 = seed1;  // Initialization & Poisson
	Initialization(seed0, seed2);

	//////////EPSP && IPSP
	//CS[0][1] = 0;
	//Exchange(neu[1], neu[0]);
	//neu[1].v -= S[0]/C;


	t0 = clock();
	if (Lyapunov)
		MLE = Largest_Lyapunov(seed2, 1, T_step);
	else
		Run_model();
	t1 = clock();

	int total_fire_num[2] = { 0 };
	for (int i = 0; i < N; i++)
		total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;

	mean_fire_rate = (total_fire_num[0] + total_fire_num[1]) / T_Max * 1000 / N; //(Hz)
	printf("mean rate (Hz) = %0.2f ", mean_fire_rate);

	printf("Total time = %0.3fs \n\n", double(t1 - t0) / CLOCKS_PER_SEC);
	Delete();


}