//-----------------------------------------------------------------------------
//		Find Cubic Hermite roots by real roots of x^3+a x^2+b x+c=0
//-----------------------------------------------------------------------------
#define SWAP(a,b) do { double tmp = b ; b = a ; a = tmp ; } while(0)

int gsl_poly_solve_cubic(double a, double b, double c,
	double &x0, double &x1, double &x2)
{
	double q = (a * a - 3 * b);
	double r = (2 * a * a * a - 9 * a * b + 27 * c);

	double Q = q / 9;
	double R = r / 54;

	double Q3 = Q * Q * Q;
	double R2 = R * R;

	double CR2 = 729 * r * r;
	double CQ3 = 2916 * q * q * q;

	if (R == 0 && Q == 0)
	{
		x0 = -a / 3;
		x1 = -a / 3;
		x2 = -a / 3;
		return 3;
	}
	else if (CR2 == CQ3)
	{
		/* this test is actually R2 == Q3, written in a form suitable
		for exact computation with integers */

		/* Due to finite precision some double roots may be missed, and
		considered to be a pair of complex roots z = x +/- epsilon i
		close to the real axis. */

		double sqrtQ = sqrt(Q);

		if (R > 0)
		{
			x0 = -2 * sqrtQ - a / 3;
			x1 = sqrtQ - a / 3;
			x2 = sqrtQ - a / 3;
		}
		else
		{
			x0 = -sqrtQ - a / 3;
			x1 = -sqrtQ - a / 3;
			x2 = 2 * sqrtQ - a / 3;
		}
		return 3;
	}
	else if (R2 < Q3)
	{
		double sgnR = (R >= 0 ? 1 : -1);
		double ratio = sgnR * sqrt(R2 / Q3);
		double theta = acos(ratio);
		double norm = -2 * sqrt(Q);
		x0 = norm * cos(theta / 3) - a / 3;
		x1 = norm * cos((theta + 2.0 * PI) / 3) - a / 3;
		x2 = norm * cos((theta - 2.0 * PI) / 3) - a / 3;

		/* Sort *x0, *x1, *x2 into increasing order */

		if (x0 > x1)
			SWAP(x0, x1);

		if (x1 > x2)
		{
			SWAP(x1, x2);

			if (x0 > x1)
				SWAP(x0, x1);
		}

		return 3;
	}
	else
	{
		double sgnR = (R >= 0 ? 1 : -1);
		double A = -sgnR * cbrt(fabs(R) + sqrt(R2 - Q3));
		double B = Q / A;
		x0 = A + B - a / 3;
		return 1;
	}
}

double cubic_hermite_real_root(double x1, double x2,  // when x2-x1 is too close to zero O(1e-6), may return -nan(ind)
	double f1, double f2,
	double df1, double df2, double rhs)
{
	// normalize to find root
	f1 -= rhs;
	f2 -= rhs;
	if (abs(f1) < 1e-12)
		return x1;
	if (abs(f2) < 1e-12)
		return x2;

	// normalize to x=[0,1]
	df1 *= x2 - x1;
	df2 *= x2 - x1;

	double c[4], s0 = NAN, s1 = NAN, s2 = NAN;
	// Coefficients for hermit interpolation:
	// c[3] x^3 + c[2] x^2 + c[1] x + c[0] = 0
	c[0] = f1;
	c[1] = df1;
	c[2] = -2 * df1 - df2 - 3 * (f1 - f2);
	c[3] = df1 + df2 + 2 * (f1 - f2);
	// TODO: check c[3] == 0
	gsl_poly_solve_cubic(c[2] / c[3], c[1] / c[3], c[0] / c[3], s0, s1, s2);
	if (0 <= s0 && s0 <= 1) {
		return s0 * (x2 - x1) + x1;
	}
	if (0 <= s1 && s1 <= 1) {
		return s1 * (x2 - x1) + x1;
	}
	if (0 <= s2 && s2 <= 1) {
		return s2 * (x2 - x1) + x1;
	}
	//fprintf(stderr, "No root in this interval\n");
	return NAN;
}