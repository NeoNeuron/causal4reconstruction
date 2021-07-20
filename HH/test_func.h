int neu_id[10] = { 88,21,29,11,23,81,80,24,44,85 };


///// state pattern, 010100010101
void record_fire_pattern(double t)
{
	double s[10];
	for (int i = 0; i < 10; i++)
		if (t - neu[neu_id[i]].last_fire_time < 10)
			s[i] = 1;
		else
			s[i] = 0;	
	fwrite(s, sizeof(double), 10, FP_fire_pattern);
}

/////// FILE name ffp for fire_pattern 
//void decide_fire_pattern_name(int trial_id)
//{
//	char str[500], c[10];
//
//	double n, m;
//	n = log(T_Step_Large) / log(0.5);
//	m = log(T_Step_Small) / log(0.5);
//
//	strcpy(str,file), strcat(str, "Pattern_"), strcat(str, "RK");
//	sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//
//	if (Lib_method == 0)
//	{
//		strcat(str, "t_l="), sprintf(c, "%0.1f", n), strcat(str, c);
//		strcat(str, "t_s="), sprintf(c, "%0.1f", m), strcat(str, c);
//	}
//	else
//	{
//		strcat(str, "lib_"), strcat(str, "t="), sprintf(c, "%0.1f", n), strcat(str, c);
//	}
//
//	strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//	strcat(str, "s="), sprintf(c, "%0.2f", S[0]), strcat(str, c);
//	strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//	if (Nu < Epsilon)
//		strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//	else
//		strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//	strcat(str, "-"), sprintf(c, "%d", trial_id), strcat(str, c), strcat(str, ".dat");
//
//	ffp = fopen(str, "wb");
//}

