#include "Kmean.h"
#include "common_header.h"
using namespace std;
void read_connect_matrix(char fname[], vector<vector<double> > &connect_matrix, vector<vector<double> > &connect_matrix_strength)
{
	double *A, *B, *C;
	FILE *fp;
	fp = fopen(fname, "rb");
	if (fp == NULL)
	{
		printf("Warning! can not open connect matrix file! %s \n",fname);
		return;
	
		//	exit(0);
	}

	long int file_size;
	double s;
	fseek(fp, 0, SEEK_END);
	file_size = ftell(fp)/sizeof(double);
	fseek(fp, 0, SEEK_SET);

	int N = connect_matrix.size();
	if (file_size >= N*N) // dense matrix
	{
		for (int i = 0; i < N; i++)		
			fread(&connect_matrix[i][0], sizeof(double), int(N), fp);	
	}
	else // sparse matrix
	{

		A = new double[file_size/3 + 1];
		fread(A, sizeof(double), N + 1, fp);
		B = new double[int(A[N])];
		C = new double[int(A[N])];
		fread(B, sizeof(double), int(A[N]), fp);
		fread(C, sizeof(double), int(A[N]), fp);
		fclose(fp);

		for (int i = 0; i < N; i++)
		{
			for (int j = int(A[i]); j < int(A[i + 1]); j++)
			{
				connect_matrix[i][int(B[j])] = 1;
				connect_matrix_strength[i][int(B[j])] = C[j];
			}
		}
		delete[]A, delete[]B, delete[]C;
	}

}

void pre_compute(int &n, vector<double>& One_D_Ma, vector<vector<double> >& Ma, int id, int L, int NE, int NI)
{
	int N = NE+NI;
	n = 0;

	int k[4];
	if (id == 0)
	{
		k[0] = 0, k[1] = N, k[2] = 0, k[3] = N;
	}
	else if (id == 1)
	{
		k[0] = 0, k[1] = NE, k[2] = 0, k[3] = NE;
	}
	else if (id == 2)
	{
		k[0] = 0, k[1] = NE, k[2] = NE, k[3] = N;
	}
	else if (id == 3)
	{
		k[0] = NE, k[1] = N, k[2] = 0, k[3] = NE;
	}
	else if (id == 4)
	{
		k[0] = NE, k[1] = N, k[2] = NE, k[3] = N;
	}
	else
	{
		printf("Wrong id=%d in pre_compute_te\n", id);
		exit(0);
	}

	for (int i = k[0]; i < k[1]; i++)
	{
		for (int j = k[2]; j < k[3]; j++)
		{
			if (Ma[i][j] > 2.0 / L) // revise
			{
				One_D_Ma[n] = log(Ma[i][j]) / log(10);
				n++;
			}
		}
	}

}

void K_mean_2(int n, vector<double>& te, double &x)    
{

	if (n == 0)
	{
		x = 1;
		return;
	}
	sort(&te[0], &te[n]);
	double mean_x, mean_y, mean_x0, mean_y0;

	mean_x = te[0];
	mean_y = te[n - 1];

	if (mean_y - mean_x <= 0.4)
	{
		x = mean_x - 1e-4;
		return;
	}

	while (1)
	{
		mean_x0 = mean_x;
		mean_y0 = mean_y;
		double s = 0, s1 = 0, a = (mean_x+mean_y)/2;
		int num = 0, num1 = 0;

		for (int i = 0; i < n; i++)
		{
			if (te[i] < a)
			{
				s += te[i];
				num++;
			}
			else
			{
				s1 += te[i];
				num1++;
			}
		}
		mean_x = s / num;
		mean_y = s1 / num1;
		if (abs(mean_x - mean_x0) + abs(mean_y - mean_y0) < 1e-8)
			break;
	}
	double a = (mean_x + mean_y) / 2;
	for (int i = 0; i < n; i++)
	{
		if (te[i] >= a)
		{
			x = te[i]-1e-4;
			break;
		}
	}
}

void compare(vector<vector<double> >& connect_matrix, vector<vector<double> >& Ma, double *x, double &Accuracy, int NE, int NI)
{
	int N = NE+NI;
	int num = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == j)
				continue;

			int id = 0;
			if (NE && NI)
			{
				if (i < NE && j < NE)
					id = 0;
				else if (i < NE && j >= NE)
					id = 1;
				else if (i >= NE && j < NE)
					id = 2;
				else if (i >= NE && j >= NE)
					id = 3;
			}

			if (Ma[i][j] < x[id] && connect_matrix[i][j] == 0)
				num++;
			else if (Ma[i][j] >= x[id] && connect_matrix[i][j] == 1)
				num++;
		}
	}
	Accuracy = 1.0*num / N / (N - 1) * 100;

}

void Kmean(vector<vector<double> >& Ma, double &Threshold, double &Accuracy, char conn_mat_fname[], int L, int NE, int NI)
{
	int N = Ma.size();
	vector<vector<double> > connect_matrix(N, vector<double>(N, 0));
	vector<vector<double> > connect_matrix_strength(N, vector<double>(N, 0));
	vector<double> One_D_Ma(N*N, 0);

	read_connect_matrix(conn_mat_fname, connect_matrix, connect_matrix_strength);
	

	// classify EE,IE,EI,II (1,2,3,4), 0--only NE or NI
	int n;
	double x[4] = {0};

	if (NE && NI)
		for (int i = 0; i < 4; i++)
		{
			pre_compute(n, One_D_Ma, Ma, i + 1, L, NE, NI);
			K_mean_2(n, One_D_Ma, x[i]);
			x[i] = pow(10, x[i]);
		}
	else
	{
		pre_compute(n, One_D_Ma, Ma, 0, L, NE, NI);
		K_mean_2(n, One_D_Ma, x[0]);
		x[0] = pow(10, x[0]);
	}

	Threshold = x[0];
	compare(connect_matrix, Ma, x, Accuracy, NE, NI);
	//printf("Ths are %0.3e %0.3e %0.3e %0.3e\n", x[0], x[1], x[2], x[3]); //revise

}
