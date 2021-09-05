#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <malloc.h>
#include <omp.h>
#include <algorithm>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace Eigen;
using namespace std;

int num_threads_openmp;

//////////////////////////////////////////
int N, NE, NI;			// total neuron number
double T_Max, DT;		// total T_Max, division into DT length
int auto_T_max = 1;
double bin;				// bin size
int order[3];			// order x , order y , max order +1
int k;					// k = order_y
double delay;			// delay: x(t+delay+bin), x(t+delay), y(t)
int tau;				//tau=delay/bin,  delay: x(n+tau+1), x(n+tau), y(n)
double L, accuracy[4], threshold[4];

char input_filename[200], output_filename[200], matrix_name[100];
char path_input[200], path_output[200];
int m[2];		// pow(2.0,order_x) and pow(2,order_y)

//////////////////////////////////////////
unsigned int **X;		// X[id][0/1]  spike information
double **z;				// z[i-->j][p] pair wise TE's p(x,x-,y-)

double **TE, **GC, **DMI, **NCC;			//TE[N][N], y-->x	
double **TE_2, **DMI_2, **NCC_2;			// 2TE,2DMI,NCC^2 [N][N], y-->x
FILE *FP;           /// output pairwise TE. (TE,y,x,px,py,p0,dp1,dp2,...,dpk,delta,te_order5, GC,sumDMI,sumNCC^2,appxo for 2sumDMI)
					////y-->x.  Total N*N*(k+12)+9, +output L,threshold*4, accuracy*4. delta = p(x=1|y=1)-p(x=1|y=0) 			

bool verbose;

/////////
#define MAX(a,b)  ((a)>(b)?(a):(b))

std::vector<int> str2vec(std::string str) {
    std::vector<int> out;
    std::stringstream ss(str);

    int tmpVal;
    while (ss >> tmpVal)
        out.push_back(tmpVal);

    return out;
}