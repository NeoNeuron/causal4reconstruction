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
	int l = order[1];	// order of y, i.e., pre-synaptic neuron
	s[3] = 0;			// p(x = 1)
	for (int i = m[0] * m[1]; i < 2 * m[0] * m[1]; i++)
		s[3] += z[i];

	s[4] = 0;			// p(y-[n] = 1)  mean firing rate of post-synaptic neuron
	for (int i = 0; i < m[0] * m[1]; i++)
		s[4] += z[2 * i + 1];

	double p0;	// p(x = 1 | x- = 0, y- = 0)
	p0 = z[m[0] * m[1]] / (z[m[0] * m[1]] + z[0]);
	s[5] = p0;

	//  p(x = 1 | x- = 0, y-[l-i] = 1) - p(x = 1 | x- = 0, y- = 0)
	for (int i = 1; i <= l; i++)
	{
		int	id_y = int(pow(2.0, l - i) + 0.01);

		s[i + 5] = z[id_y + m[0] * m[1]] / (z[id_y + m[0] * m[1]] + z[id_y]);
		s[i + 5] -= p0;
	}

	/* 
	TODO: Disable calculation of p(x- = 1|y- = 1)-p(x- = 1|y- = 0)
	// suitable for order_y = 1 or yn=1
	double p11 = 0, p10 = 0;
	for (int i = 0; i < m[0]; i++)
	{
		p11 += z[i*m[1] * 2 + m[1] + 1];
		p10 += z[i*m[1] * 2 + m[1]];
	}
	s[l + 6] = p11 / s[4] - p10 / (1 - s[4]);

	*/

    // Delta p_m := p(x = 1, y- = 1)/p(x = 1)/p(y- = 1) - 1
	double p11 = 0, py = 0;
	for (int i = 1; i <= l; i++) //order[1]
	{
		int	id_y = int(pow(2.0, l - i) + 0.01);
		for (int j = 0; j < m[0]; j++) {
			p11 += z[id_y + j * m[1] + m[0]*m[1]];
			py  += z[id_y + j * m[1] + m[0]*m[1]] + z[id_y + j * m[1]];
		}
	}
	// if (p11 == 0) {
	// 	printf("[WARNING]: p11 = %e, and py = %e\n", p11, py);
	// }
	s[l + 6] = p11 / py / s[3] - 1;


	// suitable for order_y = 1 or yn=1
	s[l + 7] = 0;
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

		//s[l + 7] += 0.5*(ss + z[i * m[1] + 1]) / p_a0 * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4]);
		s[l + 7] += 0.5*(ss + z[i * m[1] + 1]) * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4])*(1 / p_a0 + 1 / (1 - p_a0));
	}
}

double compute_TE(vector<double>& z, int *m)
{
	double H[4] = { 0 };      // -H(X,X-,Y-), -H(X-), -H(X-,Y-), -H(X,X-)  y-->x

	for (int i = 0; i < 2 * m[0] * m[1]; i++)
	{
		if (z[i] != 0)
			H[0] += z[i] * log(z[i]);
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
	int l = order[1];	// order of y, i.e., pre-synaptic neuron
	// in the order of  XX_Y_
	int n = 1 + order[0] + l;
	vector<int> zz(n, 0);
	vector<double> p(n, 0);  //p(x=1) 
	vector<double> cov_xx_(order[0], 0), cov_xx_y_(n-1, 0);
	vector<vector<double> > pxy(l, vector<double>(4, 0)); // l*4, p(x = 1, y- = 1), 10,01,00
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
		for (int j = 0; j < l; j++)
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
			if (zz[n - 1] && zz[j + l])
				cov_xx_[j] += z[i];

		// cov(x,x_y_)
		for (int j = 0; j < n - 1; j++)
			if (zz[n - 1] && zz[j])
				cov_xx_y_[j] += z[i];

		// cov(x_)
		for (int j = 0; j < order[0]; j++)
			for (int k = 0; k < order[0]; k++)
				if (zz[j + l] && zz[k + l])
					cov_x_[j][k] += z[i];

		// cov(x_y_)
		for (int j = 0; j < n - 1; j++)
			for (int k = 0; k < n - 1; k++)
				if (zz[j] && zz[k])
					cov_x_y_[j][k] += z[i];

	}

	// cov(x,x_)
	for (int j = 0; j < order[0]; j++)
	{
		cov_xx_[j] -= p[n - 1] * p[j + l];
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
		for (int k = 0; k < order[0]; k++)
		{
			cov_x_[j][k] -= p[j + l] * p[k + l];
			COV_X_(j, k) = cov_x_[j][k];
		}
	}

	// cov(x_y_)
	for (int j = 0; j < n - 1; j++)
	{
		for (int k = 0; k < n - 1; k++)
		{
			cov_x_y_[j][k] -= p[j] * p[k];
			COV_X_Y_(j, k) = cov_x_y_[j][k];
		}
	}


	// compute GC
	double aa, bb, cov_x;
	cov_x = p[n - 1] - p[n - 1] * p[n - 1];

	aa = cov_x - COV_XX_.transpose()*COV_X_.inverse()*COV_XX_;
	bb = cov_x - COV_XX_Y_.transpose()*COV_X_Y_.inverse()*COV_XX_Y_;
	s[l + 8] = log(aa / bb);

	// compute sum DMI
	double ss_xy = 0,ss_x, ss_y = 0; // H(X,Y-),H(X),H(Y-)

	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < 4; j++)
			if (pxy[i][j])
				ss_xy += pxy[i][j] * log(pxy[i][j]);

		if (p[i] > 0 && p[i] < 1)
			ss_y += p[i] * log(p[i]) + (1 - p[i]) * log(1 - p[i]);
	}
	ss_x = p[n-1] * log(p[n-1]) + (1 - p[n-1]) * log(1 - p[n-1]);

	s[l + 9] = ss_xy - l * ss_x - ss_y;

	// compute sum NCC^2
	double ss_ncc = 0;
	for (int i = 0; i < l; i++)
		ss_ncc += cov_xx_y_[i] * cov_xx_y_[i] / cov_x / cov_x_y_[i][i];
	s[l + 10] = ss_ncc;

	double ss_app = 0;
	for (int i = 0; i < l; i++)
	{
		double a;
		a = pxy[i][0] / p[i] - pxy[i][1] / (1 - p[i]);
		ss_app += a * a;
	}

	s[l + 11] = ss_app * s[4] / s[3];
}


void compute_causality(
	vector<vector<double>> &z, int *order, int *m, int N, FILE *ofile,
	bool mask_toggle, vector<vector<double>> &mask_indices)
{	
	int l = order[1]; // order of y (presynaptic neuron)

	int num_pairs = mask_toggle ? mask_indices[0].size() : N * N;
	for (int id = 0; id < num_pairs; id++)  // i-->j
	{
		double *s;
		s = new double[l + 12];
		for (int i = 0; i < l + 12; i++)
			s[i] = 0;

		int pre_idx, post_idx;
		if (mask_toggle)
			pre_idx = mask_indices[0][id], post_idx = mask_indices[1][id];
		else
			pre_idx = id / N, post_idx = id % N;
		s[1] = pre_idx;
		s[2] = post_idx;

		// Comment below the calculate auto-correlation
		if (pre_idx == post_idx) {
			fwrite(s, sizeof(double), l + 12, ofile);
			continue;
		}

		s[0] = compute_TE(z[id], m); 
		compute_s(z[id], order, m, s);
		compute_GC_sum_DMI_NCC(z[id], order, m, s);

		fwrite(s, sizeof(double), l + 12, ofile);

		delete[]s;
	}
}



