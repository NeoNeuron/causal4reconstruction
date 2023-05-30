#ifndef __KMEAN__
#define __KMEAN__
#include <vector>
using namespace std;

void read_connect_matrix(vector<vector<double> > &connect_matrix, vector<vector<double> > &connect_matrix_strength);

void pre_compute(int &n, vector<double>& One_D_Ma, vector<vector<double> >& Ma, int id, int L, int NE, int NI);

void K_mean_2(int n, vector<double>& te, double &x);

void compare(vector<vector<double> >& connect_matrix, vector<vector<double> >& Ma, double *x, double &Accuracy, int NE, int NI);

void Kmean(vector<vector<double> >& Ma, double &Threshold, double &Accuracy, char conn_mat_fname[], int L, int NE, int NI);

#endif // !__KMEAN__