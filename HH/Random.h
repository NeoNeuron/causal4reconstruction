//-----------------------------------------------------------------------------
//		Linear generater of the uniform distribution in U£¨0, 1£©
//-----------------------------------------------------------------------------
const unsigned int Random_M = (unsigned int)(pow(2.0, 31));

double Random(long &X)
{
	int x;
	double r;	
	x = X % Random_M;
	if(x<0)
		x=Random_M+x;
	r = (x+0.0)/Random_M;
	X = (314159269*x+453806245)%Random_M;
	return r;
}