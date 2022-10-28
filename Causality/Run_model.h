void Print(double **a, char str[])
{
	printf("%s:\n", str);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("%0.3e ", a[i][j]);
		printf("\n");
	}printf("\n");
}

void Find_T_Max(FILE *fp)
{
	double s[2];
	while (!feof(fp))
	{
		fread(s, sizeof(double), 2, fp);
	}
	if (verbose)
		printf("T_max=%0.3e\n", s[0]);
	T_Max = s[0];
	rewind(fp);
}

// read data in (t0, t1]
void read_data(FILE *fp, int data_length, double t0, double t1)
{
	double s[2];
	int id, id_t;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < data_length; j++)
			X[i][j] = 0;

	while (!feof(fp))
	{
		if (fread(s, sizeof(double), 2, fp) != 2)
			break;
		if (s[0] > t0 && s[0] <= t1 && s[1] < N)
		{
			id = int(s[1]);
			id_t = int((s[0] - t0) / bin);
			X[id][id_t] = 1;
		}
		if (s[0] > t1)
			break;
	}
	//	rewind(fp);
}

void compute_p(int data_length, int y, int x)  // y-->x  XX-Y- 101
{
	int data_num;

	int tid_x = order[2] - 1 + tau;
	int tid_y = order[2] - 1 - 1;

	for (int i = 0; i < data_length - (order[2] - 1 + tau); i++)
	{
		// Translate binary spike sequence to decimal represetation.
		int x_coding, y_coding;

		x_coding = X[x][tid_x];
		y_coding = X[y][tid_y];

		for (int j = 1; j < order[0] + 1; j++)
			x_coding = x_coding * 2 + X[x][tid_x - j];
		for (int j = 1; j < order[1]; j++)
			y_coding = y_coding * 2 + X[y][tid_y - j];
		z[y*N + x][x_coding*m[1] + y_coding] += 1.0;

		tid_x++;
		tid_y++;
	}
}


void Run_model()
{
	num_threads_openmp = N >= 8 ? 8 : 4;

	int read_repeat, data_length, interval;
	char str[200];
	FILE *fp;
	clock_t t0, t1, t2, t3;

	t0 = clock();

	strcpy(str, path_input), strcat(str, input_filename), strcat(str, "_spike_train.dat");
	fp = fopen(str, "rb");
	if (fp == NULL)
	{
		printf("Error in read file: %s\n%s\n", path_input, input_filename);
		exit(0);
	}

	//	 compute the probability distribution	
	if(auto_T_max)
		Find_T_Max(fp); // find the data time T_Max

	if (T_Max / DT <= 0.99)
	{
		printf("Warning! T_max=%0.3e < DT=%0.3e!\n", T_Max, DT);
		printf("path:%s\nfilename:%s\n", path_input, input_filename);
		T_Max = DT;
	}
	read_repeat = int(T_Max / DT + 0.01);

	data_length = int(DT / bin);

	X = new unsigned int*[N];
	for (int i = 0; i < N; i++)
		X[i] = new unsigned int[data_length+1];

	Create_Matrix();

	for (int id = 0; id < read_repeat; id++)
	{
		read_data(fp, data_length, 1.0*id*data_length*bin, 1.0*(1 + id)*data_length*bin);
		//Notice!  id*data_length may larger than MAX INT 2147483647

		// shuffling data
		if (shuffle_flag) {
			for (int neu_id=0; neu_id < N; neu_id ++)
				shuffle(&X[neu_id][0], &X[neu_id][data_length], rng);
		}
		
#pragma omp parallel for num_threads(num_threads_openmp)

		for (int i = 0; i < N*N; i++)  // y-->x
		{
			int x, y;
			y = i / N;
			x = i % N;

			// Comment below the calculate auto-correlation
			if (y == x)
				continue;
			compute_p(data_length, y, x);
		}
	}
	fclose(fp);

	for (int i = 0; i < N; i++)
		delete[]X[i];	
	delete[]X;

	
	L = 0;
	for (int i = 0; i < 2 * m[0] * m[1]; i++)
		L += z[1][i];

	for (int i = 0; i < N*N; i++)
		for (int j = 0; j < 2 * m[0] * m[1]; j++)
			z[i][j] /= L;

	// compute TE(x_n+1+tau,x_n+tau,y_n), GC(x_n+1+tau,x_n+tau,y_n) 
	// sum DMI(x_n+1+tau,y_n), CC(x_n+1+tau,y_n)

	FP = fopen(output_filename, "wb");
	if (verbose)
		printf("save to file :%s\n\n", output_filename);
	compute_causality();

	char cch[4][100] = { {"GC"},{"sum NCC"},{"2 sum DMI"},{"2TE"} };

	if (N <= 10) //revise
	{
		if (verbose) {
			Print(TE_2, cch[3]);

			//Print(TE_2, cch[3]);
			//Print(GC, cch[0]);
			//Print(DMI_2, cch[2]);
			//Print(NCC_2, cch[1]);
		}
	}

	//  reconstruction & kmean	
	Kmean(TE_2, threshold[0], accuracy[0]);
	Kmean(GC, threshold[1], accuracy[1]);
	Kmean(DMI_2, threshold[2], accuracy[2]);
	Kmean(NCC_2, threshold[3], accuracy[3]);
	fwrite(&L, sizeof(double), 1, FP);
	fwrite(&threshold, sizeof(double), 4, FP);
	fwrite(&accuracy, sizeof(double), 4, FP);


	fclose(FP);
	Delete_Matrix();



	t1 = clock();
	if (verbose) {
		printf("T_max=%0.2e L=%0.2e\n", T_Max, L);

		for (int i = 0; i < 4; i++)
			printf("th=%0.3e ", threshold[i]);
		printf("\n");

		for (int i = 0; i < 4; i++)
			printf("accu=%0.2f ", accuracy[i]);
		printf("\n");

		printf("Total time=%0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
	}
}