#include "common_header.h"
#include "Run_model.h"
#include <vector>
using namespace std;

void Print(vector<vector<double> >& a, char str[])
{
	printf("%s:\n", str);
	for (auto i = a.begin(); i != a.end(); i++)
	{
		for (auto j = (*i).begin(); j < (*i).end(); j++)
			printf("%0.3e ", *j);
		printf("\n");
	}
	printf("\n");
}

double Find_T_Max(FILE *fp, bool verbose)
{
	double s[2];
	while (!feof(fp))
		fread(s, sizeof(double), 2, fp);
	if (verbose)
		printf("T_max=%0.3e\n", s[0]);
	rewind(fp);
	return s[0];
}

// read data in [t0, t1)
void read_data(
	FILE *fp, double t0, double t1, double bin, 
	vector<vector<unsigned int> >& X, int N)
{
	double s[2];
	int node_id, time_id;

	while (!feof(fp))
	{
		if (fread(s, sizeof(double), 2, fp) != 2)
			break;
		if (s[0] >= t0 && s[0] < t1 && s[1] < N)
		{
			node_id = int(s[1]);
			time_id = int((s[0] - t0) / bin);
			X[node_id][time_id] = 1;
		}
		if (s[0] >= t1)
			break;
	}
	//	rewind(fp);
}

// y-->x  XX-Y- 101
void compute_p(
	vector<vector<unsigned int> >& X, int y, int x, int N,
	vector<vector<double> >& z, int tau, int *order, int *m)
{
	for (int i = order[2] - 1 + tau; i < X[0].size(); i++)
	{
		// Translate binary spike sequence to decimal represetation.
		int x_coding, y_coding;

		x_coding = X[x][i];
		y_coding = X[y][i - tau - 1];

		for (int j = 1; j < order[0] + 1; j++)
			x_coding = x_coding * 2 + X[x][i - j];
		for (int j = 1; j < order[1]; j++)
			y_coding = y_coding * 2 + X[y][i - tau - 1 - j];
		z[y*N + x][x_coding*m[1] + y_coding] += 1.0;
	}
}
