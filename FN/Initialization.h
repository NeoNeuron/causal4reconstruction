double Decide_S(int i, int j, long &seed, long &seed1)  // i-->j
{
	double s;

	if (i < NE && j < NE)
		s = S[0];  //scaling 
	else if (i < NE && j >= NE)
		s = S[1];
	else if (i >= NE && j < NE)
		s = S[2];
	else
		s = S[3];

	if (random_S == 0)
		return s;
	else if (random_S == 1)						// uniform [0, 2s]
		return Random(seed) * 2 * s;
	else if (random_S == 2)						// gauss N(s,(s/4)^2)
		return abs(sqrt(-2 * log(Random(seed)))*cos(2 * PI*Random(seed1))*s / 4 + s);
	else if (random_S == 3)                     // Exponential E(s)
		return -log(1 - Random(seed)) * s;
	else		//// Log normal sigma=0.794 mu=log(s)-sigma^2/2.  Std/Mean=0.93  X~exp(mu+sigma*Z),Z~N(0,1)
	{
		double sigma, mu;
		sigma = 0.794, mu = log(s) - sigma * sigma / 2;
		double b = sqrt(-2 * log(Random(seed)))*cos(2 * PI*Random(seed1))*sigma + mu;
		return exp(b);
	}
}

double Decide_Nu(int i, long &seed, long &seed1)  // i-->j
{
	double s = Nu;
	if (random_Nu == 0)
		return s;
	else if (random_Nu == 1)					 // uniform [0, 2s]
		return Random(seed) * 2 * s;
	else if (random_Nu == 2)					 // gauss N(s,(s/4)^2)
		return abs(sqrt(-2 * log(Random(seed)))*cos(2 * PI*Random(seed1))*s / 4 + s);
	else if (random_Nu == 3)                     // Exponential E(s)
		return -log(1 - Random(seed)) * s;
	else										 // Log normal, log(X)~N(mu,sigma^2)
	{
		double a = log(s);
		double b = sqrt(-2 * log(Random(seed)))*cos(2 * PI*Random(seed1))*sqrt(-a / 2) + 1.25*a;
		return exp(b);
	}


}

void Create_connect_matrix(long& seed)
{
	Connect_Matrix = new double *[N];
	for (int i = 0; i < N; i++)
		Connect_Matrix[i] = new double[N]{0};

	CS = new double *[N];
	for (int i = 0; i < N; i++)
		CS[i] = new double[N]{0};
	long seed1 = 15, seed2 = 43;

	for (int i = 0; i < 1000; i++)
	{
		Random(seed1), Random(seed2);
	}

	if (N == 3)
	{
		Connect_Matrix[0][1] = 1;
		Connect_Matrix[1][2] = 1;

		CS[0][1] = S[0];		 ///////scaling	
		CS[1][2] = S[0];
	}
	else if( N == 4 || N == 5)
	{
		Connect_Matrix[0][1] = 1;
		Connect_Matrix[1][2] = 1;
		Connect_Matrix[0][3] = 1;

		CS[0][1] = S[0];
		CS[1][2] = S[0];
		CS[0][3] = S[0];
	}
	else
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (i != j && Random(seed) < P_c)
				{
					Connect_Matrix[i][j] = 1;
					CS[i][j] = Decide_S(i, j, seed1, seed2);
				}
			}
		}
	}

	if (record_data[0] || record_data[1])
	{
		FILE *fp;
		char str[200], ch[10];

		strcpy(str, file), strcat(str, "connect_matrix-p=");
		sprintf(ch, "%0.3f", P_c), strcat(str, ch);

		if (random_S == 1)
			strcat(str, "-U");
		else if (random_S == 2)
			strcat(str, "-G");
		else if (random_S == 3)
			strcat(str, "-E");
		else if (random_S == 4)
			strcat(str, "-LN");
		strcat(str, ".dat");
		
		if ((fp = fopen(str, "rb")) == NULL)
		{
			fp = fopen(str, "wb");
			for (int i= 0; i < N; i++)
				fwrite(Connect_Matrix[i], sizeof(double), N, fp);

			if (random_S != 0)
				for (int i = 0; i < N; i++)
					fwrite(CS[i], sizeof(double), N, fp);
			fclose(fp);
		}

	}
}



void Initialization(long &seed0,long &seed2)
{
	Create_connect_matrix(seed0);
	neu = new struct neuron[N];
	neu_old = new struct neuron[N];

	long Seed = 11, Seed1 = 43;
	for (int i = 0; i < 1000; i++)
	{
		Random(Seed), Random(Seed1);
	}

	for (int i = 0; i < N; i++)
	{
		neu[i].t = 0;
		neu[i].Nu = Decide_Nu(i, Seed, Seed1);


		neu[i].v = -1.199408;
		neu[i].dv = 0;
		neu[i].w = -0.624260;	

		neu[i].I_input = 0;
		neu[i].fire_num = 0;
		neu[i].last_fire_time = -1e5;
		neu[i].if_fired = 0;
		neu[i].Poisson_input_time = new double[int(T_step*Nu * 2) + 5];
		for (int j = 0; j < 500; j++)
			Random(seed2);
		neu[i].seed = seed2;
		neu[i].Poisson_input_num = -1;

		neu[i].wait_strength_E = 0;
		neu[i].wait_strength_I = 0;

		neu[i].state = 1;
		neu_old[i].state = 0;
	}
	
}
