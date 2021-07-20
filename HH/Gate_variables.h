//-----------------------------------------------------------------------------
//		Functions of gate variables
//-----------------------------------------------------------------------------
//#define alpha_m(v) (abs((v)+40.0) < 1e-7 ? 1+0.5*((v)+40)/10: 0.1*((v)+40)/(1-exp(-((v)+40)/10)))
//#define alpha_h(v) (0.07*exp(-((v)+65)/20))
//#define alpha_n(v) (abs((v)+55.0) < 1e-7 ? 0.1+0.05*((v)+55)/10 : 0.01*((v)+55)/(1-exp(-((v)+55)/10)))
//#define beta_m(v) (4*exp(-((v)+65)/18))
//#define beta_h(v) (1/(1+exp(-(35+(v))/10)))
//#define beta_n(v) (0.125*exp(-((v)+65)/80)) 

double alpha_m(double v)
{
	double s = 0.1*(v + 40);
	if (abs(s) < 1e-6)
		return 1 + s / 2 + s*s / 12;
	else
		return s / (1 - exp(-s));
}

double alpha_h(double v)
{
	return 0.07*exp(-(v + 65) / 20);
}

double alpha_n(double v)
{
	double s = 0.1*(v + 55);
	if (abs(s) < 1e-6)
		return 0.1*(1 + s / 2 + s*s / 12);
	else
		return 0.1*s / (1 - exp(-s));
}

double beta_m(double v)
{
	return 4 * exp(-(v + 65) / 18);
}

double beta_h(double v)
{
	return 1.0 / (1 + exp(-(35 + v) / 10));
}

double beta_n(double v)
{
	return 0.125*exp(-(v + 65) / 80);
}




