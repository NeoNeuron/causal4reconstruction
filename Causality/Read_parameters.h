void Read_parameters()
{
	FILE *fp;

	//if ((fp = fopen("/home/shangj/TE_base_IF/parameters_input_TE.txt", "r")) == NULL)
	//	fp = fopen("D:/code/TE_base_IF/parameters_input_TE.txt", "r");

	//if (fp == NULL)
	//	fp = fopen("/home/zqtian/TE_base_IF/parameters_input_TE.txt", "r");


	if ((fp = fopen("./NetCau_parameters.txt", "r")) == NULL)
		fp = fopen("./NetCau_parameters.txt", "r");

	if (fp == NULL)
		fp = fopen("./NetCau_parameters.txt", "r");


	if (fp == NULL)
	{
		printf("Error in Read_parameters()! :: Cann't open parameters input file! \n");
		getchar();// system("pause");
		exit(1);
	}

	char ch[100];

	fscanf(fp, "%s%d%s%d", ch, &NE, ch, &NI);
	N = NE + NI;
	fscanf(fp, "%s%lf%lf", ch, &T_Max, &DT);
	if (DT <= 1e5)
		DT = 1e5;
	fscanf(fp, "%s%d", ch, &auto_T_max);


	fscanf(fp, "%s%lf", ch, &bin);
	fscanf(fp, "%s%d%d", ch, &order[0], &order[1]);  //  order 0--x, 1--y

	fscanf(fp, "%s%lf", ch, &delay);
	//int aa;
	//fscanf(fp, "%s%d", ch, &aa);

	fscanf(fp, "%s%s", ch, input_filename);
	fscanf(fp, "%s%s", ch, matrix_name);
	fscanf(fp, "%s%s", ch, path_input);
	fscanf(fp, "%s%s", ch, path_output);

	if (N == NE)
	{
		strcat(path_input, "EE/N=");
		strcat(path_output, "EE/N=");
	}
	else if (N == NI)
	{
		strcat(path_input, "II/N=");
		strcat(path_output, "II/N=");
	}
	else
	{
		strcat(path_input, "EI/N=");
		strcat(path_output, "EI/N=");
	}

	sprintf(ch, "%d", N), strcat(path_input, ch), strcat(path_input, "/");
	sprintf(ch, "%d", N), strcat(path_output, ch), strcat(path_output, "/");

	fclose(fp);
}

void Output_filename()
{
	tau = int(delay / bin);
	k = order[1];
	m[0] = int(pow(2.0, order[0]));
	m[1] = int(pow(2.0, order[1]));
	order[2] = order[0] > order[1] ? order[0] : order[1];
	order[2]++;

	char str[200], ch[20];

	strcpy(str, path_output);

	strcat(str, "TGIC2.0-K="), sprintf(ch, "%d", order[0]), strcat(str, ch);
	strcat(str, "_"), sprintf(ch, "%d", order[1]), strcat(str, ch);

	strcat(str, "bin="), sprintf(ch, "%0.2f", bin), strcat(str, ch);
	strcat(str, "delay="), sprintf(ch, "%0.2f", delay), strcat(str, ch);
	strcat(str, "-"), strcat(str, input_filename);

	//strcat(str, "T="), sprintf(ch, "%0.0e", T_Max), strcat(str, ch);
	//strcat(str, "T"), sprintf(ch, "%d", int(T_Max / DT + 0.5)), strcat(str, ch);
	strcat(str, ".dat");

	strcpy(output_filename, str);
}