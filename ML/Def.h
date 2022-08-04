#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <malloc.h>
#include <algorithm>
#include <omp.h>

using namespace std;
#define PI (atan(1)*4)
int num_threads_openmp;
#define Epsilon 1e-10

//-----------------------------------------------------------------------------
//		Parameters with fixed value 
//-----------------------------------------------------------------------------
// //////class II
//#define G_Ca 4.4
//#define E_Ca 120.0
//#define G_K 8.0
//#define E_K -80.0
//#define G_L 2.0
//#define E_L -60.0	
//#define C 20.0
//#define V1  -1.2
//#define V2	18.0
//#define V3	2.0
//#define V4	30.0
//#define phi 0.04

//////class I
#define G_Ca 4.0
#define E_Ca 120.0
#define G_K 8.0
#define E_K -80.0
#define G_L 2.0
#define E_L -60.0	
#define C 20.0
#define V1 -1.2
#define V2	18.0
#define V3	12.0
#define V4	17.4
#define phi (1.0/15)


#define V_th  0.0
#define T_ref 25.0


//-----------------------------------------------------------------------------
//		Variables for input parameters
//-----------------------------------------------------------------------------						
int N, NE, NI;                // Number of neurons(total,excitatory,inhibirory)
double T_Max, T_step;         // Total time & time step
double S[4];                  // Coupling strength (E->E,E->I,I->E,I->I) 
int I_CONST;                  // electrode current constant  
double I_const_input;         // constant input current
double Nu, f[2];                // Feedforward Poisson rate and strength(E,I)
int random_S, random_Nu;
double P_c;                   // Connect probability 
int Lyapunov;                     // compute largest lyapunov exponnet
int Power_spectrum = 0;				  // record v for power spectrum	
int record_data[2];				// save data or not
char file[200];				  // Record data path
double Record_v_start, Record_v_end;

//-----------------------------------------------------------------------------
//		Netwrok Information
//-----------------------------------------------------------------------------
double **Connect_Matrix;         // Connect matrix
double **CS;					 // Coupling strength matrix

struct neuron
{
	double t, Nu;
	double v, dv, n;
	double I_input;
	double last_fire_time;
	double fire_num;
	int if_fired;

	double *Poisson_input_time;
	int Poisson_input_num;
	long seed;
	double wait_strength_E, wait_strength_I;
	int state;     //1--neu,0--neu_old
};
struct neuron *neu, *neu_old;


#define MIN(a,b)  ((a)<(b)?(a):(b))
#define MAX(a,b)  ((a)>(b)?(a):(b))
//-----------------------------------------------------------------------------
//		Record firing time and voltage
//-----------------------------------------------------------------------------
FILE *FP, *FP1, *FP_FFTW, *FP_fire_pattern;
FILE *ffp; // for test

