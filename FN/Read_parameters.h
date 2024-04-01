void Read_parameters(long &seed, long &seed1)
{
	FILE *fp;
	fp = fopen("./NetModel_parameters.txt", "r");
	if (fp == NULL)
		fp = fopen("./FN/NetModel_parameters.txt", "r");

	if (fp == NULL)
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
	fscanf(fp, "%s%lf%lf%lf%lf", ch, &S[0], &S[1], &S[2], &S[3]);
	fscanf(fp, "%s%d", ch, &I_CONST);

	fscanf(fp, "%s%lf%s%lf", ch, &Nu, ch, &f[0]);
	f[1] = f[0];

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
}


void out_put_filename()
{
	char str[200] = "", c[10], str1[200];

	if (Lyapunov)
	{
		record_data[0] = 0;
		record_data[1] = 0;
	}

	strcpy(str, "FN");

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
		strcat(str, "U");
	else if (random_Nu == 2)
		strcat(str, "G");
	else if (random_Nu == 3)
		strcat(str, "E");
	else if (random_Nu == 4)
		strcat(str, "LN");

	printf("dt=%0.3f, T_Max=%0.2e\n", T_step, T_Max);

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

	if (record_data[0] || record_data[1])
	{
		if (NE == N)
			printf("file:NE=%d\\%s\n", N, str);
		else if (NI == N)
			printf("file:NI=%d\\%s\n", N, str);
		else
			printf("file:NEI=%d\\%s\n", N, str);
	}
	
}