#ifndef __COMPUTE_CAUSALITY_
#define __COMPUTE_CAUSALITY_

#include <vector>
#include <stdio.h>
using namespace std;

void compute_s(vector<double>& z, int *order, int *m, double *s);

double compute_TE(vector<double>& z, int *m);

// GC, sum DMI(x_n+1+tau,y_n^(l)) & NCC^2(x_n+1+tau,y_n^(l)) & appro for 2sumDMI
void compute_GC_sum_DMI_NCC(vector<double>& z, int *order, int *m, double *s);

void compute_causality(
    vector<vector<double>> &z, int *order, int *m, int N, FILE *ofile,
	vector<vector<double>> &TE, vector<vector<double>> &GC, 
	vector<vector<double>> &DMI, vector<vector<double>> &NCC, 
	vector<vector<double>> &TE_2, vector<vector<double>> &DMI_2);

#endif // !__COMPUTE_CAUSALITY_
