void compute_s(double *s, int id)
{
	// TE y-->x
	// TE,y,x,px,py,p0,dp1,dp2,...,dpk, delta,  te_order5  
	//  px = p(x=1), py = p(y=1), delta = p(x=1|y=1)-p(x=1|y=0) 	or delta = p(x=1|y=1)-p(x=1|y=0)

	/*
	if (k == 1)
	{
	double a, b, c, p1, p1_, p2_;  // p1 = 1/p1_, 1-p1 = 1/p2_
	a = z[id][0] + z[id][4];
	b = z[id][1] + z[id][5];
	c = b / (a + b);
	p1 = z[id][4] / a;
	p1_ = 1 / p1, p2_ = 1 / (1 - p1);

	s[3] = z[id][4] + z[id][5] + z[id][6] + z[id][7];
	s[4] = z[id][1] + z[id][3] + z[id][5] + z[id][7];

	s[5] = z[id][4]/ (z[id][0] + z[id][4]);

	s[6] = z[id][5] / b - p1;

	s[7] = (z[id][3] + z[id][7]) / s[4];
	s[7] -= (z[id][2] + z[id][6]) / (1-s[4]);

	s[8] = (1 - c)*p1_*p2_ - s[6] / 3 * (p1_*p1_ - p2_*p2_)*(1 - c*c) + s[6] * s[6] / 6 * (p1_*p1_*p1_ + p2_*p2_*p2_)*(1 - c*c*c)\
	- s[6] * s[6] * s[6] / 10 * (p1_*p1_*p1_*p1_ - p2_*p2_*p2_*p2_)*(1 - c*c*c*c);
	s[8] *= s[6] * s[6] * b / 2;
	}
	else if (k == 2)
	{
	double a, b, c, d, e, p1, p1_, p2_;

	a = z[id][0] + z[id][16];
	b = z[id][1] + z[id][17];
	c = z[id][2] + z[id][18];
	p1 = z[id][16] / a;
	p1_ = 1 / p1, p2_ = 1 / (1 - p1);

	s[3] = 0;
	for (int l = 16; l < 32; l++)
	s[3] += z[id][l];

	s[4] = 0;
	for (int l = 1; l < 32; l += 2)
	s[4] += z[id][l];

	s[5] = z[id][16] / (z[id][16] + z[id][0]);

	s[6] = z[id][18] / c - p1;
	s[7] = z[id][17] / b - p1;

	s[8] = (z[id][5] + z[id][21]) / s[4];
	s[8] -= (z[id][4] + z[id][20]) / (1-2*s[4]);


	d = b*s[7] + c*s[6];
	e = a + b + c;

	s[9] = 0.5*(p1_ + p2_)*(b*s[7] * s[7] + c*s[6] * s[6] - d*d / e) \
	- 1.0 / 6 * (p1_*p1_ - p2_*p2_)*(b*s[7] * s[7] * s[7] + c*s[6] * s[6] * s[6] - d*d*d / e / e)\
	+ 1.0 / 12 * (p1_*p1_*p1_ + p2_*p2_*p2_)*(b*s[7] * s[7] * s[7] * s[7] + c*s[6] * s[6] * s[6] * s[6] - d*d*d*d / e / e / e)\
	- 1.0 / 20 * (p1_*p1_*p1_*p1_ - p2_*p2_*p2_*p2_)*(b*s[7] * s[7] * s[7] * s[7] * s[7] + c*s[6] * s[6] * s[6] * s[6] * s[6] - d*d*d*d*d / e / e / e / e);
	}
	else
	{
	s[3] = 0;
	for (int l = m[0]*m[1]; l < 2* m[0] * m[1]; l++)
	s[3] += z[id][l];

	s[4] = 0;
	for (int l = 0; l < m[0] * m[1]; l++)
	s[4] += z[id][2 * l + 1];

	double p0;
	p0 = z[id][m[0] * m[1]] / (z[id][m[0] * m[1]]+z[id][0]);
	s[5] = p0;

	for (int l = 1; l <= order[1]; l++)
	{
	int id_y = int(pow(2.0, l - l)+0.01);
	s[l + 5] = z[id][id_y+ m[0] * m[1]]/(z[id][id_y + m[0] * m[1]]+z[id][id_y]);
	s[l + 5] -= p0;
	}

	////// suitable for order_y = 1 or yn=1
	double p11 = 0, p10 = 0;
	for (int l = 0; l < m[0]; l++)
	{
	p11 += z[id][l*m[1] * 2 + m[1] + 1];
	p10 += z[id][l*m[1] * 2 + m[1]];
	}
	s[k + 6] = p11 / s[4] - p10 / (1 - s[4]);

	s[k + 7] = 0;

	////// suitable for order_y = 1 or yn=1
	for (int i = 0; i < m[0]; i++)
	{
	double p_a0, p_a1, ss;

	ss = z[id][m[1] * m[0] + i * m[1] + 0];
	if (ss > 0)
	p_a0 = ss / (ss + z[id][i * m[1]]);
	else
	{
	p_a0 = 0;
	continue;
	}

	ss = z[id][m[1] * m[0] + i * m[1] + 1];
	if (ss > 0)
	p_a1 = ss / (ss + z[id][i * m[1] + 1]);
	else
	p_a1 = 0;

	//s[k + 7] += 0.5*(ss + z[id][i * m[1] + 1]) / p_a0 * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4]);
	s[k + 7] += 0.5*(ss + z[id][i * m[1] + 1]) * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4])*(1 / p_a0 + 1 / (1 - p_a0));
	}
	}
	*/

	s[3] = 0;	// p(x = 1)
	for (int l = m[0] * m[1]; l < 2 * m[0] * m[1]; l++)
		s[3] += z[id][l];

	s[4] = 0;   // p(y-[l] = 1)
	for (int l = 0; l < m[0] * m[1]; l++)
		s[4] += z[id][2 * l + 1];

	double p0;
	p0 = z[id][m[0] * m[1]] / (z[id][m[0] * m[1]] + z[id][0]);
	s[5] = p0;

	for (int l = 1; l <= order[1]; l++)
	{
		int	id_y = int(pow(2.0, order[1] - l) + 0.01);

		s[l + 5] = z[id][id_y + m[0] * m[1]] / (z[id][id_y + m[0] * m[1]] + z[id][id_y]);
		s[l + 5] -= p0;
	}


	/* TODO: Disable calculation of p(x- = 1|y- = 1)-p(x- = 1|y- = 0)
	// suitable for order_y = 1 or yn=1
	double p11 = 0, p10 = 0;
	for (int l = 0; l < m[0]; l++)
	{
		p11 += z[id][l*m[1] * 2 + m[1] + 1];
		p10 += z[id][l*m[1] * 2 + m[1]];
	}
	s[k + 6] = p11 / s[4] - p10 / (1 - s[4]);

	*/

	double p11 = 0, py = 0;
	for (int l = 1; l <= 1; l++) //order[1]
	{
		int	id_y = int(pow(2.0, order[1] - l) + 0.01);
		for (int i = 0; i < m[0]; i++) {
			p11 += z[id][id_y + i * m[1] + m[0]*m[1]];
			py  += z[id][id_y + i * m[1] + m[0]*m[1]] + z[id][id_y + i * m[1]];
		}
	}
	// if (p11 == 0) {
	// 	printf("[WARNING]: p11 = %e, and py = %e\n", p11, py);
	// }
	s[k + 6] = p11 / py / s[3] - 1;

	s[k + 7] = 0;

	// suitable for order_y = 1 or yn=1
	for (int i = 0; i < m[0]; i++)
	{
		double p_a0, p_a1, ss;

		ss = z[id][m[1] * m[0] + i * m[1] + 0];
		if (ss > 0)
			p_a0 = ss / (ss + z[id][i * m[1]]);
		else
		{
			p_a0 = 0;
			continue;
		}

		ss = z[id][m[1] * m[0] + i * m[1] + 1];
		if (ss > 0)
			p_a1 = ss / (ss + z[id][i * m[1] + 1]);
		else
			p_a1 = 0;

		//s[k + 7] += 0.5*(ss + z[id][i * m[1] + 1]) / p_a0 * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4]);
		s[k + 7] += 0.5*(ss + z[id][i * m[1] + 1]) * (p_a1 - p_a0)*(p_a1 - p_a0)*(1 - s[4])*(1 / p_a0 + 1 / (1 - p_a0));
	}
}

double compute_TE(int id)
{
	double H[4] = { 0 };      // -H(X,X-,Y-), -H(X-), -H(X-,Y-), -H(X,X-)  y-->x

	for (int l = 0; l < 2 * m[0] * m[1]; l++)
	{
		if (z[id][l] != 0)
			H[0] += z[id][l] * log(z[id][l]);
	}

	for (int id_x_ = 0; id_x_ < m[0]; id_x_++)
	{
		double p = 0;
		for (int id_x = 0; id_x < 2; id_x++)
			for (int id_y_ = 0; id_y_ < m[1]; id_y_++)
			{
				int id_p = (id_x*m[0] + id_x_)*m[1] + id_y_;
				p += z[id][id_p];
			}
		if (p != 0)
			H[1] += p*log(p);
	}

	for (int id_x_y_ = 0; id_x_y_ < m[0] * m[1]; id_x_y_++)
	{
		double p = z[id][id_x_y_] + z[id][id_x_y_ + m[0] * m[1]];
		if (p != 0)
			H[2] += p*log(p);
	}

	for (int id_xx_ = 0; id_xx_ < 2 * m[0]; id_xx_++)
	{
		double p = 0;
		for (int id_y_ = 0; id_y_ < m[1]; id_y_++)
			p += z[id][id_xx_*m[1] + id_y_];

		if (p != 0)
			H[3] += p*log(p);
	}
	return  H[0] + H[1] - H[2] - H[3];
}

//double compute_GC_old(int id, double *s)  
//{
//	double px = s[3], py = s[4];
//	double a, b, c, d, e;
//	double S_xx_ , S_xx_y_;
//	S_xx_ = px - px*px - px*px*px / (1 - px);
//
//	/////// cov(x-oy-) = (a,b;b,c)
//	/////// cov(x,x-oy-)=(d,e)
//	a = px - px*px;
//	b = 0;
//	for (int id_x = 0; id_x < m[0]; id_x++)
//		for (int id_y = 2; id_y < m[1]; id_y++)
//			b += z[id][id_x*m[1] * 2 + id_y * 2 + m[1] + 1];
//	c = py - py*py;
//	d = -px*px;
//	e = 0;
//	for (int id_x = 0; id_x < m[0]; id_x++)
//		for (int id_y = 0; id_y < m[1] / 2; id_y++)
//			e += z[id][id_x*m[1] + id_y + m[0] * m[1] + m[1] / 2];
//	e -= px*py;
//
//	S_xx_y_ = a - (c*d*d + a*e*e - 2 * b*d*e) / (a*c - b*b);
//
//	if (id == 1)
//	{
//		printf("old aa=%e bb=%e %e\n", S_xx_, S_xx_y_, log(S_xx_ / S_xx_y_));
//	}
//
//	return log(S_xx_ / S_xx_y_);
//}

//void compute_GC(double *s, int id)
//{
//	//// in the order of  XX_Y_
//	int n = 1 + order[0] + order[1];
//	int *zz = new int[n];
//	double *p; //p(x=1), 
//	double *cov_xx_, *cov_xx_y_;
//	double **cov_x_, **cov_x_y_;
//	MatrixXd COV_X_(order[0], order[0]), COV_X_Y_(n - 1, n - 1);
//	VectorXd COV_XX_(order[0]), COV_XX_Y_(n - 1);
//
//
//	p = new double[n];
//	cov_xx_ = new double[order[0]];
//	cov_xx_y_ = new double[n - 1];
//
//	cov_x_ = new double *[order[0]];
//	for (int i = 0; i < order[0]; i++)
//		cov_x_[i] = new double[order[0]];
//	
//	cov_x_y_ = new double *[n - 1];
//	for (int i = 0; i < n - 1; i++)
//		cov_x_y_[i] = new double[n - 1];
//
//	////// initialize
//	for (int i = 0; i < n; i++)
//		p[i] = 0;
//	for (int i = 0; i < order[0]; i++)
//		cov_xx_[i] = 0;
//
//	for (int i = 0; i < n - 1; i++)
//		cov_xx_y_[i] = 0;
//
//	for (int i = 0; i < order[0]; i++)
//		for (int j = 0; j < order[0]; j++)
//			cov_x_[i][j] = 0;
//
//	for (int i = 0; i < n - 1; i++)
//		for (int j = 0; j < n - 1; j++)
//			cov_x_y_[i][j] = 0;
//
//
//	for (int i = 0; i < 2 * m[0] * m[1]; i++)
//	{
//		if (z[id][i] == 0)
//			continue;
//
//		int a = i;
//		for (int j = 0; j < n; j++)  //zz_n,zz_n-1,...,zz_1
//		{
//			zz[j] = a % 2;
//			a /= 2;
//		}
//
//		////p
//		for (int j = 0; j < n; j++)
//			if (zz[j])
//				p[j] += z[id][i];
//
//		//// cov(x,x_)
//		for (int j = 0; j < order[0]; j++)
//			if (zz[n - 1] && zz[j + order[1]])
//				cov_xx_[j] += z[id][i];
//
//		//// cov(x,x_y_)
//		for (int j = 0; j < n - 1; j++)
//			if (zz[n - 1] && zz[j])
//				cov_xx_y_[j] += z[id][i];
//
//		/// cov(x_)
//		for (int j = 0; j < order[0]; j++)
//			for (int l = 0; l < order[0]; l++)
//				if (zz[j + order[1]] && zz[l + order[1]])
//					cov_x_[j][l] += z[id][i];
//
//		/// cov(x_y_)
//		for (int j = 0; j < n - 1; j++)
//			for (int l = 0; l < n - 1; l++)
//				if (zz[j] && zz[l])
//					cov_x_y_[j][l] += z[id][i];
//
//	}
//
//	//// cov(x,x_)
//	for (int j = 0; j < order[0]; j++)
//	{
//		cov_xx_[j] -= p[n - 1] * p[j + order[1]];
//		COV_XX_(j) = cov_xx_[j];
//	}
//
//	//// cov(x,x_y_)
//	for (int j = 0; j < n - 1; j++)
//	{
//		cov_xx_y_[j] -= p[n - 1] * p[j];
//		COV_XX_Y_(j) = cov_xx_y_[j];
//	}
//
//	/// cov(x_)
//	for (int j = 0; j < order[0]; j++)
//	{
//		for (int l = 0; l < order[0]; l++)
//		{
//			cov_x_[j][l] -= p[j + order[1]] * p[l + order[1]];
//			COV_X_(j, l) = cov_x_[j][l];
//		}
//	}
//
//	/// cov(x_y_)
//	for (int j = 0; j < n - 1; j++)
//	{
//		for (int l = 0; l < n - 1; l++)
//		{
//			cov_x_y_[j][l] -= p[j] * p[l];
//			COV_X_Y_(j, l) = cov_x_y_[j][l];
//		}
//	}
//
//	double aa, bb;
//	aa = p[n - 1] - p[n - 1] * p[n - 1] - COV_XX_.transpose()*COV_X_.inverse()*COV_XX_;
//	bb = p[n - 1] - p[n - 1] * p[n - 1] - COV_XX_Y_.transpose()*COV_X_Y_.inverse()*COV_XX_Y_;
//
//	s[k + 8] = log(aa / bb);
//
//	delete[]zz, delete[]p, delete[]cov_xx_, delete[] cov_xx_y_;
//	for (int i = 0; i < order[0]; i++)
//		delete[] cov_x_[i];
//	delete[] cov_x_;
//
//	for (int i = 0; i < n - 1; i++)
//		delete[] cov_x_y_[i];
//	delete[] cov_x_y_;
//}
// 
////////DMI(x_n+1+tau,y_n) & NCC(x_n+1+tau,y_n) & dp in DMI& NCC
//void compute_DMI_NCC(double *s, int id) 
//{
//	double p[4] = {0};
//	int Id[4];
//	double H[3] = { 0 };   // -H(X,Y-),-H(X),H(Y-)
//
//
//	Id[0] = m[0] * m[1] + m[1] / 2; //x_n+1+tau=1,y_n=1
//	Id[1] = m[0] * m[1];			//x_n+1+tau=1,y_n=0
//	Id[2] = m[1] / 2;				//x_n+1+tau=0,y_n=1
//	Id[3] = 0;						//x_n+1+tau=0,y_n=0
//
//	for (int i = 0; i < 4; i++)
//	{
//		for (int id_x_ = 0; id_x_ < m[0]; id_x_++)
//			for (int id_y = 0; id_y < m[1] / 2; id_y++)
//				p[i] += z[id][Id[i] + id_x_ * m[1] + id_y];
//
//		if (p[i])
//			H[0] += p[i] * log(p[i]);
//	}
//
//	double px, py;
//	px = p[0] + p[1];
//	py = p[0] + p[2];
//	
//	if (px < 1 && px>0)
//		H[1] = px * log(px) + (1 - px)*log(1 - px);
//
//	if (py < 1 && py>0)
//		H[2] = py * log(py) + (1 - py)*log(1 - py);
//
//	s[k + 9] = H[0] - H[1] - H[2];	///DMI
//
//	s[k + 10] = (p[0] - px * py) / sqrt(px - px * px) / sqrt(py-py*py);  /// NCC
//
//	s[k + 11] = p[0] / py - p[1] / (1 - py);  ///dp in DMI& NCC
//}


////// GC, sum DMI(x_n+1+tau,y_n^(l)) & NCC^2(x_n+1+tau,y_n^(l)) & appro for 2sumDMI
void compute_GC_sum_DMI_NCC(double *s, int id)
{
	//// in the order of  XX_Y_
	int n = 1 + order[0] + order[1];
	int *zz = new int[n];
	double *p;  //p(x=1) 
	double *cov_xx_, *cov_xx_y_;
	double **pxy; // order[1]*4, p(x = 1, y- = 1), 10,01,00
	double **cov_x_, **cov_x_y_;
	MatrixXd COV_X_(order[0], order[0]), COV_X_Y_(n - 1, n - 1);
	VectorXd COV_XX_(order[0]), COV_XX_Y_(n - 1);



	p = new double[n];
	cov_xx_ = new double[order[0]];
	cov_xx_y_ = new double[n - 1];

	pxy = new double *[order[1]];
	for (int i = 0; i < order[1]; i++)
		pxy[i] = new double[4]; 

	cov_x_ = new double *[order[0]];
	for (int i = 0; i < order[0]; i++)
		cov_x_[i] = new double[order[0]];

	cov_x_y_ = new double *[n - 1];
	for (int i = 0; i < n - 1; i++)
		cov_x_y_[i] = new double[n - 1];

	////// initialize
	for (int i = 0; i < n; i++)
		p[i] = 0;
	for (int i = 0; i < order[0]; i++)
		cov_xx_[i] = 0;
	for (int i = 0; i < n - 1; i++)
		cov_xx_y_[i] = 0;

	for (int i = 0; i < order[1]; i++)
		for (int j = 0; j < 4; j++)
			pxy[i][j] = 0;

	for (int i = 0; i < order[0]; i++)
		for (int j = 0; j < order[0]; j++)
			cov_x_[i][j] = 0;

	for (int i = 0; i < n - 1; i++)
		for (int j = 0; j < n - 1; j++)
			cov_x_y_[i][j] = 0;


	for (int i = 0; i < 2 * m[0] * m[1]; i++)
	{
		if (z[id][i] == 0)
			continue;

		int a = i;
		for (int j = 0; j < n; j++)  //zz_n,zz_n-1,...,zz_1
		{
			zz[j] = a % 2;
			a /= 2;
		}

		////p
		for (int j = 0; j < n; j++)
			if (zz[j])
				p[j] += z[id][i];

		////pxy
		for (int j = 0; j < order[1]; j++)
		{
			if (zz[n - 1] == 1 && zz[j] == 1)
				pxy[j][0] += z[id][i];
			if (zz[n - 1] == 1 && zz[j] == 0)
				pxy[j][1] += z[id][i];
			if (zz[n - 1] == 0 && zz[j] == 1)
				pxy[j][2] += z[id][i];
			if (zz[n - 1] == 0 && zz[j] == 0)
				pxy[j][3] += z[id][i];
		}

		//// cov(x,x_)
		for (int j = 0; j < order[0]; j++)
			if (zz[n - 1] && zz[j + order[1]])
				cov_xx_[j] += z[id][i];

		//// cov(x,x_y_)
		for (int j = 0; j < n - 1; j++)
			if (zz[n - 1] && zz[j])
				cov_xx_y_[j] += z[id][i];

		/// cov(x_)
		for (int j = 0; j < order[0]; j++)
			for (int l = 0; l < order[0]; l++)
				if (zz[j + order[1]] && zz[l + order[1]])
					cov_x_[j][l] += z[id][i];

		/// cov(x_y_)
		for (int j = 0; j < n - 1; j++)
			for (int l = 0; l < n - 1; l++)
				if (zz[j] && zz[l])
					cov_x_y_[j][l] += z[id][i];

	}

	//// cov(x,x_)
	for (int j = 0; j < order[0]; j++)
	{
		cov_xx_[j] -= p[n - 1] * p[j + order[1]];
		COV_XX_(j) = cov_xx_[j];
	}

	//// cov(x,x_y_)
	for (int j = 0; j < n - 1; j++)
	{
		cov_xx_y_[j] -= p[n - 1] * p[j];
		COV_XX_Y_(j) = cov_xx_y_[j];
	}

	/// cov(x_)
	for (int j = 0; j < order[0]; j++)
	{
		for (int l = 0; l < order[0]; l++)
		{
			cov_x_[j][l] -= p[j + order[1]] * p[l + order[1]];
			COV_X_(j, l) = cov_x_[j][l];
		}
	}

	/// cov(x_y_)
	for (int j = 0; j < n - 1; j++)
	{
		for (int l = 0; l < n - 1; l++)
		{
			cov_x_y_[j][l] -= p[j] * p[l];
			COV_X_Y_(j, l) = cov_x_y_[j][l];
		}
	}


	//// compute GC
	double aa, bb, cov_x;
	cov_x = p[n - 1] - p[n - 1] * p[n - 1];

	aa = cov_x - COV_XX_.transpose()*COV_X_.inverse()*COV_XX_;
	bb = cov_x - COV_XX_Y_.transpose()*COV_X_Y_.inverse()*COV_XX_Y_;
	s[k + 8] = log(aa / bb);

	/////compute sum DMI
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

	////compute sum NCC^2
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


	///// delete
	delete[]zz, delete[]p, delete[]cov_xx_, delete[] cov_xx_y_;

	for (int i = 0; i < order[1]; i++)
		delete[] pxy[i];
	delete[] pxy;

	for (int i = 0; i < order[0]; i++)
		delete[] cov_x_[i];
	delete[] cov_x_;

	for (int i = 0; i < n - 1; i++)
		delete[] cov_x_y_[i];
	delete[] cov_x_y_;



}


void compute_causality()
{	
	double *s;        // TE,y,x,px,py,p0,dp1,dp2,...,dpk, delta,  te_order5, GC, sumDMI, sumNCC^2, appro for 2sumDMI  
					  //  px = p(x=1), py = p(y=1), delta = p(x = 1, y- = 1)/p(x = 1)/p(y- = 1) - 1, dp = p(x=1|y-=1)-p(x=1|y-=0)	
	int id;
	s = new double[k + 12];    

	for (int i = 0; i < N; i++)  // i-->j
	{
		for (int j = 0; j < N; j++)
		{

			for (int l = 0; l < k + 12; l++)
				s[l] = 0;

			id = i * N + j;
			s[1] = i;
			s[2] = j;

			// Comment below the calculate auto-correlation
			if (i == j)
			{
				fwrite(s, sizeof(double), k + 12, FP);
				TE[i][j] = 0, GC[i][j] = 0, DMI[i][j] = 0, NCC[i][j] = 0;
				TE_2[i][j] = 0, DMI_2[i][j] = 0, NCC_2[i][j] = 0;
				continue;
			}




			s[0] = compute_TE(id), compute_s(s, id);

			//compute_GC(s, id);
			//compute_DMI_NCC(s, id);

			compute_GC_sum_DMI_NCC(s, id);

			fwrite(s, sizeof(double), k + 12, FP);



			TE[i][j] = s[0], GC[i][j] = s[k + 8];
			DMI[i][j] = s[k + 9], NCC[i][j] = s[k + 10];

			TE_2[i][j] = TE[i][j] * 2;
			DMI_2[i][j] = DMI[i][j] * 2;
			NCC_2[i][j] = NCC[i][j];


		}
	}

	delete[]s;
}



