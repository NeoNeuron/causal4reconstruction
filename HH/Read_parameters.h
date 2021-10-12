#include "mkdir.h"

void Read_parameters(long &seed, long &seed1)
{
	FILE *fp;
	fp = fopen("NetModel_parameters.txt", "r");

	if(fp == NULL)
		fp = fopen("./HH/NetModel_parameters.txt", "r");

	if(fp == NULL)
	{
		printf("Error in Read_parameters()! :: Cann't open parameters input file! \n");
		getchar();// system("pause");
		exit(1);
	}

	char ch[100];

	fscanf(fp, "%s%d%s%d", ch, &NE, ch, &NI);
	N = NE + NI;
	fscanf(fp, "%s%ld%ld", ch, &seed, &seed1);

	fscanf(fp, "%s%lf%s%lf", ch, &T_Max, ch, &T_step);
	fscanf(fp, "%s", ch);
	for (int i=0; i<4; i++)
		fscanf(fp, "%lf", &S[i]);
	fscanf(fp, "%s%d", ch, &I_CONST);

	// Nu
	fscanf(fp, "%s%lf", ch, &Nu);
	// f
	fscanf(fp, "%s", ch);
	f = new double[N];
	for (int i=0; i<N; i++) {
		fscanf(fp, "%lf", &f[i]);
		printf("%f", f[i]);
	}

	// Create the read the connect_matrix
	fscanf(fp, "%s", ch);
	Connect_Matrix = new double *[N];
	for (int i = 0; i < N; i++) {
		Connect_Matrix[i] = new double[N];
		for (int j = 0; j < N; j++)
			fscanf(fp, "%lf", &Connect_Matrix[i][j]);
	}

	fscanf(fp, "%s%lf", ch, &P_c);
	fscanf(fp, "%s%d", ch, &random_S);
	if (random_S > 4 || random_S < 0)
	{
		printf("Error random_S=%d\n", random_S);
		getchar();
		exit(0);
	}
	while (fgetc(fp) != '\n');

	fscanf(fp, "%s%d", ch, &random_Nu);
	if (random_Nu > 4 || random_Nu < 0)
	{
		printf("Error random_Nu=%d\n", random_Nu);
		getchar();
		exit(0);
	}
	while (fgetc(fp) != '\n');


	fscanf(fp, "%s%d", ch, &Lyapunov);
	fscanf(fp, "%s%d%d", ch, &record_data[0], &record_data[1]);
	fscanf(fp, "%s%lf%lf", ch, &Record_v_start, &Record_v_end);
	fscanf(fp, "%s%s", ch, file);
	fclose(fp);

	if (N == NE)
		strcat(file, "EE/N=");
	else if (N == NI)
		strcat(file, "II/N=");
	else
		strcat(file, "EI/N=");
	sprintf(ch, "%d", N), strcat(file, ch), strcat(file, "/");

	// initialize folder
	_mkdir(file);
}


void out_put_filename()
{
	ode_type = 2;
	IntOrd = 1;
	method = 3;
	T_step = 0.2;
	Power_spectrum = 0;
	Estimate_RK4_call = 0;
	RecordFP = 0;
	if (fabs(T_step - int(T_step)) < 1e-8)
		T_step = pow(0.5, T_step);
	
	S_d_e = exp(-T_step / Sigma_d_E);
	S_d_i = exp(-T_step / Sigma_d_I);
	S_r_e = exp(-T_step / Sigma_r_E);
	S_r_i = exp(-T_step / Sigma_r_I);

	Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;

	if (method <= 1)
		Regular_method = 1;
	else if (method == 2)
		Lib_method = 1;
	else if (method == 3)
		ETDRK_method = 1;
	else if (method == 4)
		ETD_method = 1;

	char str[200] = "", c[10], str1[200];

	if (Lyapunov)
	{
		record_data[0] = 0;
		record_data[1] = 0;
	}

	if (CP < 1e-6)
		strcpy(str, "HH");
	else
	{
		strcpy(str, "P"), sprintf(c, "%0.2f", CP), strcat(str, c);
		strcat(str, "HH");
	}

	if (random_S == 1)
		strcat(str, "U-");
	else if (random_S == 2)
		strcat(str, "G-");
	else if (random_S == 3)
		strcat(str, "E-");
	else if (random_S == 4)
		strcat(str, "LN-");

	strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
	strcat(str, "s="), sprintf(c, "%0.3f", S[0]), strcat(str, c);  
	if (NE && NI)
	{
		strcat(str, "s="), sprintf(c, "%0.3f", S[2]), strcat(str, c);
	}
	strcat(str, "f="), sprintf(c, "%0.3f", f[0]), strcat(str, c);
	strcat(str, "u="), sprintf(c, "%0.3f", Nu), strcat(str, c);
	

	if (random_Nu == 1)
		strcat(str, "-NuU");
	else if (random_Nu == 2)
		strcat(str, "-NuG");
	else if (random_Nu == 3)
		strcat(str, "-NuE");
	else if (random_Nu == 4)
		strcat(str, "-NuLN");

	strcpy(filename, str);
	printf("T_Max=%0.2e dt=%0.3f\n", T_Max,T_step);

	if (TrialID)  // for multiple trials with fixed CS and  Poisson seeds 
	{
		strcat(str, "-"), sprintf(c, "%d", TrialID), strcat(str, c);
	}

	if (record_data[0])
	{
		strcpy(str1, file), strcat(str1, str), strcat(str1, "_spike_train.dat");
		FP = fopen(str1, "wb");
	}
	if (record_data[1])
	{
		strcpy(str1, file), strcat(str1, str), strcat(str1, "_voltage.dat");
		FP1 = fopen(str1, "wb");
	}

	if (Power_spectrum)				
	{
		char ch[200];
		strcpy(ch, file), strcat(ch, "fftw_"), strcat(ch, str), strcat(ch, ".dat");
		FP_FFTW = fopen(ch, "wb");
	}

	if (RecordFP)
	{
		if (N < 100)
		{
			printf("N=%d should >= 100 to record FP!\n", N);
			exit(0);
		}
		char ch[200];
		strcpy(ch, file), strcat(ch, "FP_"), strcat(ch, str), strcat(ch, ".dat");
		FP_fire_pattern = fopen(ch, "wb");
	}

	if (record_data[0] || record_data[1] || Power_spectrum || RecordFP)
	{
		if (NE == N)
			printf("file:NE=%d\\%s\n", N, str);
		else if (NI == N)
			printf("file:NI=%d\\%s\n", N, str);
		else
			printf("file:NEI=%d\\%s\n", N, str);
	}
	
}