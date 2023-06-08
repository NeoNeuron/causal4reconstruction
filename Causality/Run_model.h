#ifndef __RUN_MODEL_
#define __RUN_MODEL_
#include <vector>
using namespace std;

void Print(vector<double>& a, char str[], int count_in_line);

double Find_T_Max(FILE *fp, bool verbose);

// read data in [t0, t1)
void read_data(
    FILE *fp, double t0, double t1, 
    double bin, vector<vector<unsigned short int> >& X, int N);

// y-->x  XX-Y- 101
void compute_p(
    vector<vector<unsigned short int> >& X, int y, int x, int N,
    vector<vector<double> >& z, int tau, int *order, int *m,
    int z_index);

#endif // !__RUN_MODEL_