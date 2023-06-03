#include "common_header.h"
#include "Compute_Causality.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <vector>
using namespace Eigen;
using namespace std;

void compute_s(vector<double>& z, int *order, int *m, double *s)
{
	// z := p(x, x-, y-)
	int k = order[1];	// order of pre-synaptic neuron
	s[3] = 0;			// p(x = 1)
	for (int l = m[0] * m[1]; l < 2 * m[0] * m[1]; l++)
		s[3] += z[l];

	s[4] = 0;			// p(y-[n] = 1)  mean firing rate of post-synaptic neuron
	for (int l = 0; l < m[0] * m[1]; l++)
		s[4] += z[2 * l + 1];

	double p0;	// p(x = 1 | x- = 0, y- = 0)
	p0 = z[m[0] * m[1]] / (z[m[0] * m[1]] + z[0]);
	s[5] = p0;

	//  p(x = 1 | x- = 0, y-[k-i] = 1) - p(x = 1 | x- = 0, y- = 0)
	for (int l = 1; l <= order[1]; l++)
	{
		int	id_y = int(pow(2.0, order[1] - l) + 0.01);

		s[l + 5] = z[id_y + m[0] * m[1]] / (z[id_y + m[0] * m[1]] + z[id_y]);
		s[l + 5] -= p0;
	}

	/* 
	TODO: Disable calculation of p(x- = 1|y- = 1)-p(x- = 1|y- = 0)
	// suitable for order_y = 1 or yn=1
	double p11 = 0, p10 = 0;
	for (int l = 0; l < m[0]; l++)
	{
		p11 += z[l*m[1] * 2 + m[1] + 1];
		p10 += z[l*m[1] * 2 + m[1]];
	}
	s[k + 6] = p11 / s[4] - p10 / (1 - s[4]);

	*/

    // Delta p_m := p(x = 1, y- = 1)/p(x = 1)/p(y- = 1) - 1
	double p11 = 0, py = 0;
	for (int l = 1; l <= order[1]; l++) //order[1]
	{
		int	id_y = int(pow(2.0, order[1] - l) + 0.01);
		for (int i = 0; i < m[0]; i++) {
			p11 += z[id_y + i * m[1] + m[0]*m[1]];
			py  += z[id_y + i * m[1] + m[0]*m[1]] + z[id_y + i * m[1]];
		}
	}
	// if (p11 == 0) {
	// 	printf("[WARNING]: p11 = %e, and py = %e\n", p11, py);
	// }
	s[k + 6] = p11 / py / s[3] - 1;


	// suitable for order_y = 1 or yn=1
	s[k + 7] = 0;
	for (int i = 0; i < m[0]; i++)
	{
		double p_a0, p_a1, ss;

		ss = z[m[1] * m[0] + i * m[1] + 0];
		if (ss > 0)
			p_a0 = ss / (ss + z[i * m[1]]);
		else
		{
			p_a0 = 0;
			continue;
		}

		ss = z[m[1] * m[0] + i * m[1] + 1];
		if (ss > 0)
			p_a1 = ss / (ss + z[i * m[1] + 1]);
		else
			p_a1 = 0;

		//s[k + 7] += 0.5*(ss + z[i * m[1] + 1]) / p_a0 * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4]);
		s[k + 7] += 0.5*(ss + z[i * m[1] + 1]) * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4])*(1 / p_a0 + 1 / (1 - p_a0));
	}
}

double compute_TE(vector<double>& z, int *m)
{
	double H[4] = { 0 };      // -H(X,X-,Y-), -H(X-), -H(X-,Y-), -H(X,X-)  y-->x

	for (int l = 0; l < 2 * m[0] * m[1]; l++)
	{
		if (z[l] != 0)
			H[0] += z[l] * log(z[l]);
	}

	for (int id_x_ = 0; id_x_ < m[0]; id_x_++)
	{
		double p = 0;
		for (int id_x = 0; id_x < 2; id_x++)
			for (int id_y_ = 0; id_y_ < m[1]; id_y_++)
			{
				int id_p = (id_x*m[0] + id_x_)*m[1] + id_y_;
				p += z[id_p];
			}
		if (p != 0)
			H[1] += p*log(p);
	}

	for (int id_x_y_ = 0; id_x_y_ < m[0] * m[1]; id_x_y_++)
	{
		double p = z[id_x_y_] + z[id_x_y_ + m[0] * m[1]];
		if (p != 0)
			H[2] += p*log(p);
	}

	for (int id_xx_ = 0; id_xx_ < 2 * m[0]; id_xx_++)
	{
		double p = 0;
		for (int id_y_ = 0; id_y_ < m[1]; id_y_++)
			p += z[id_xx_*m[1] + id_y_];

		if (p != 0)
			H[3] += p*log(p);
	}
	return  H[0] + H[1] - H[2] - H[3];
}

// GC, sum DMI(x_n+1+tau,y_n^(l)) & NCC^2(x_n+1+tau,y_n^(l)) & appro for 2sumDMI
void compute_GC_sum_DMI_NCC(vector<double>& z, int *order, int *m, double *s)
{
	int k = order[1];
	// in the order of  XX_Y_
	int n = 1 + order[0] + order[1];
	vector<int> zz(n, 0);
	vector<double> p(n, 0);  //p(x=1) 
	vector<double> cov_xx_(order[0], 0), cov_xx_y_(n-1, 0);
	vector<vector<double> > pxy(order[1], vector<double>(4, 0)); // order[1]*4, p(x = 1, y- = 1), 10,01,00
	vector<vector<double> > cov_x_(order[0], vector<double>(order[0], 0));
	vector<vector<double> > cov_x_y_(n-1, vector<double>(n-1, 0));
	MatrixXd COV_X_(order[0], order[0]), COV_X_Y_(n - 1, n - 1);
	VectorXd COV_XX_(order[0]), COV_XX_Y_(n - 1);

	for (int i = 0; i < 2 * m[0] * m[1]; i++)
	{
		if (z[i] == 0)
			continue;

		int a = i;
		for (int j = 0; j < n; j++)  //zz_n,zz_n-1,...,zz_1
		{
			zz[j] = a % 2;
			a /= 2;
		}

		// p
		for (int j = 0; j < n; j++)
			if (zz[j])
				p[j] += z[i];

		// pxy
		for (int j = 0; j < order[1]; j++)
		{
			if (zz[n - 1] == 1 && zz[j] == 1)
				pxy[j][0] += z[i];
			if (zz[n - 1] == 1 && zz[j] == 0)
				pxy[j][1] += z[i];
			if (zz[n - 1] == 0 && zz[j] == 1)
				pxy[j][2] += z[i];
			if (zz[n - 1] == 0 && zz[j] == 0)
				pxy[j][3] += z[i];
		}

		// cov(x,x_)
		for (int j = 0; j < order[0]; j++)
			if (zz[n - 1] && zz[j + order[1]])
				cov_xx_[j] += z[i];

		// cov(x,x_y_)
		for (int j = 0; j < n - 1; j++)
			if (zz[n - 1] && zz[j])
				cov_xx_y_[j] += z[i];

		// cov(x_)
		for (int j = 0; j < order[0]; j++)
			for (int l = 0; l < order[0]; l++)
				if (zz[j + order[1]] && zz[l + order[1]])
					cov_x_[j][l] += z[i];

		// cov(x_y_)
		for (int j = 0; j < n - 1; j++)
			for (int l = 0; l < n - 1; l++)
				if (zz[j] && zz[l])
					cov_x_y_[j][l] += z[i];

	}

	// cov(x,x_)
	for (int j = 0; j < order[0]; j++)
	{
		cov_xx_[j] -= p[n - 1] * p[j + order[1]];
		COV_XX_(j) = cov_xx_[j];
	}

	// cov(x,x_y_)
	for (int j = 0; j < n - 1; j++)
	{
		cov_xx_y_[j] -= p[n - 1] * p[j];
		COV_XX_Y_(j) = cov_xx_y_[j];
	}

	// cov(x_)
	for (int j = 0; j < order[0]; j++)
	{
		for (int l = 0; l < order[0]; l++)
		{
			cov_x_[j][l] -= p[j + order[1]] * p[l + order[1]];
			COV_X_(j, l) = cov_x_[j][l];
		}
	}

	// cov(x_y_)
	for (int j = 0; j < n - 1; j++)
	{
		for (int l = 0; l < n - 1; l++)
		{
			cov_x_y_[j][l] -= p[j] * p[l];
			COV_X_Y_(j, l) = cov_x_y_[j][l];
		}
	}


	// compute GC
	double aa, bb, cov_x;
	cov_x = p[n - 1] - p[n - 1] * p[n - 1];

	aa = cov_x - COV_XX_.transpose()*COV_X_.inverse()*COV_XX_;
	bb = cov_x - COV_XX_Y_.transpose()*COV_X_Y_.inverse()*COV_XX_Y_;
	s[k + 8] = log(aa / bb);

	// compute sum DMI
	double ss_xy = 0,ss_x, ss_y = 0; // H(X,Y-),H(X),H(Y-)

	for (int i = 0; i < order[1]; i++)
	{
		for (int j = 0; j < 4; j++)
			if (pxy[i][j])
				ss_xy += pxy[i][j] * log(pxy[i][j]);

		if (p[i] > 0 && p[i] < 1)
			ss_y += p[i] * log(p[i]) + (1 - p[i]) * log(1 - p[i]);
	}
	ss_x = p[n-1] * log(p[n-1]) + (1 - p[n-1]) * log(1 - p[n-1]);

	s[k + 9] = ss_xy - order[1] * ss_x - ss_y;

	// compute sum NCC^2
	double ss_ncc = 0;
	for (int i = 0; i < order[1]; i++)
		ss_ncc += cov_xx_y_[i] * cov_xx_y_[i] / cov_x / cov_x_y_[i][i];
	s[k + 10] = ss_ncc;

	double ss_app = 0;
	for (int i = 0; i < order[1]; i++)
	{
		double a;
		a = pxy[i][0] / p[i] - pxy[i][1] / (1 - p[i]);
		ss_app += a * a;
	}

	s[k + 11] = ss_app * s[4] / s[3];
}


void compute_causality(
	vector<vector<double>> &z, int *order, int *m, int N, FILE *ofile,
	vector<vector<double>> &TE, vector<vector<double>> &GC, 
	vector<vector<double>> &DMI, vector<vector<double>> &NCC, 
	vector<vector<double>> &TE_2, vector<vector<double>> &DMI_2, 
	bool mask_toggle, vector<vector<double>> &mask_indices)
{	
	int k = order[1]; // order of y (presynaptic neuron)

	if (mask_toggle) {
		// #pragma omp parallel for num_threads(num_threads_openmp)
		for (int id = 0; id < mask_indices[0].size(); id++)  // i-->j
		{
			double *s;
			s = new double[k + 12];    
			for (int l = 0; l < k + 12; l++)
				s[l] = 0;

			int i = mask_indices[0][id], j = mask_indices[1][id];
			s[1] = i;
			s[2] = j;
			int z_id = i * N + j;

			// Comment below the calculate auto-correlation
			if (i == j)
			{
				fwrite(s, sizeof(double), k + 12, ofile);
				TE[i][j] = 0, GC[i][j] = 0, DMI[i][j] = 0, NCC[i][j] = 0;
				TE_2[i][j] = 0, DMI_2[i][j] = 0;
				continue;
			}

			s[0] = compute_TE(z[z_id], m); 
			compute_s(z[z_id], order, m, s);
			compute_GC_sum_DMI_NCC(z[z_id], order, m, s);

			fwrite(s, sizeof(double), k + 12, ofile);

			TE[i][j] = s[0], GC[i][j] = s[k + 8];
			DMI[i][j] = s[k + 9], NCC[i][j] = s[k + 10];

			TE_2[i][j] = TE[i][j] * 2;
			DMI_2[i][j] = DMI[i][j] * 2;
			delete[]s;
		}
	} else {
		// #pragma omp parallel for num_threads(num_threads_openmp)
		for (int id = 0; id < N*N; id++)  // i-->j
		{
			double *s;
			s = new double[k + 12];    
			for (int l = 0; l < k + 12; l++)
				s[l] = 0;

			int i = id / N, j = id % N;
			s[1] = i;
			s[2] = j;

			// Comment below the calculate auto-correlation
			if (i == j)
			{
				fwrite(s, sizeof(double), k + 12, ofile);
				TE[i][j] = 0, GC[i][j] = 0, DMI[i][j] = 0, NCC[i][j] = 0;
				TE_2[i][j] = 0, DMI_2[i][j] = 0;
				continue;
			}


			s[0] = compute_TE(z[id], m); 
			compute_s(z[id], order, m, s);

			//compute_GC(s, id);
			//compute_DMI_NCC(s, id);

			compute_GC_sum_DMI_NCC(z[id], order, m, s);

			fwrite(s, sizeof(double), k + 12, ofile);

			TE[i][j] = s[0], GC[i][j] = s[k + 8];
			DMI[i][j] = s[k + 9], NCC[i][j] = s[k + 10];

			TE_2[i][j] = TE[i][j] * 2;
			DMI_2[i][j] = DMI[i][j] * 2;
			delete[]s;
		}
	}
}



