void read_connect_matrix(double **connect_matrix, double **connect_matrix_strength)
{
	double *A, *B, *C;
	char str[200];
	FILE *fp;

	//strcpy(str, path_input), strcat(str, matrix_name);

	strcpy(str, path_input);
	int id_p1 = 0, id_p2 = 0;
	for (int i = 0; i < strlen(input_filename); i++)
	{
		if (input_filename[i] == 'p')
			id_p1 = i;
		if (input_filename[i] == 's' && i - id_p1 < 8)
			id_p2 = i;
	}
	strcpy(matrix_name, "connect_matrix-");
	char chh[10] = "";
	for (int i = id_p1; i < id_p2; i++)
		chh[i - id_p1] = input_filename[i];
	strcat(matrix_name, chh), strcat(matrix_name, "0.dat");
	strcat(str, matrix_name);



	fp = fopen(str, "rb");
	if (fp == NULL)
	{
		printf("Warning! can not open connect matrix file! %s \n",str);

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				connect_matrix[i][j] = 0;
		return;
	
		//	exit(0);
	}

	int num = 0;
	double s;
	while (!feof(fp))
	{
		fread(&s, sizeof(double), 1, fp);
		num++;
	}
	rewind(fp);

	if (num >= N*N + 1) // dense matrix
	{
		for (int i = 0; i < N; i++)		
			fread(connect_matrix[i], sizeof(double), int(N), fp);	
	}
	else // sparse matrix
	{

		A = new double[N + 1];
		fread(A, sizeof(double), N + 1, fp);
		B = new double[int(A[N])];
		C = new double[int(A[N])];
		fread(B, sizeof(double), int(A[N]), fp);
		fread(C, sizeof(double), int(A[N]), fp);
		fclose(fp);


		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				connect_matrix[i][j] = 0;
				connect_matrix_strength[i][j] = 0;
			}

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

void pre_compute(int &n, double *One_D_Ma, double **Ma, int id)
{
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

void K_mean_2(int n, double *te, double &x)    
{

	if (n == 0)
	{
		x = 1;
		return;
	}
	sort(te, te + n);
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

void compare(double **connect_matrix, double **Ma, double *x, double &Accuracy)
{
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

void Kmean(double **Ma, double &Threshold, double &Accuracy)
{
	double **connect_matrix, **connect_matrix_strength, *One_D_Ma;
	char str[200];

	connect_matrix = new double *[N];
	for (int i = 0; i < N; i++)
		connect_matrix[i] = new double[N];

	connect_matrix_strength = new double *[N];
	for (int i = 0; i < N; i++)
		connect_matrix_strength[i] = new double[N];

	read_connect_matrix(connect_matrix, connect_matrix_strength);
	

	// classify EE,IE,EI,II (1,2,3,4), 0--only NE or NI
	int n;
	double x[4] = {0};
	One_D_Ma = new double[N*N];

	if (NE && NI)
		for (int i = 0; i < 4; i++)
		{
			pre_compute(n, One_D_Ma, Ma, i + 1);
			K_mean_2(n, One_D_Ma, x[i]);
			x[i] = pow(10, x[i]);
		}
	else
	{
		pre_compute(n, One_D_Ma, Ma, 0);
		K_mean_2(n, One_D_Ma, x[0]);
		x[0] = pow(10, x[0]);
	}

	Threshold = x[0];
	compare(connect_matrix, Ma, x, Accuracy);
	//printf("Ths are %0.3e %0.3e %0.3e %0.3e\n", x[0], x[1], x[2], x[3]); //revise

	for (int i = 0; i < N; i++)
	{
		delete[]connect_matrix[i];
		delete[]connect_matrix_strength[i];
	}
	delete[]connect_matrix,delete[]connect_matrix_strength;
	delete[]One_D_Ma;
}
