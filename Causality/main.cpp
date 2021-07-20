/* TE based on IF model spike trains*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------



#define _CRT_SECURE_NO_WARNINGS
#include "def.h"
#include "Read_parameters.h"
#include "New_delete.h"
#include "Kmean.h" 
#include "Compute_Causality.h"
#include "test_function.h"
#include "Run_model.h"


int main(int argc, char **argv)
{
	clock_t t0, t1;
	Read_parameters();

	if (argc >= 2)
		strcpy(input_filename, argv[1]);


	Output_filename();

	t0 = clock();
	Run_model();
	t1 = clock();
	printf("Total time=%0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);

}

//int main(int argc, char **argv)          // scan delay in HH,Lorenz,FN,Hn
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//	strcpy(input_filename, argv[1]);
//	double dt = atof(argv[2]);
//
//	for (int ss = atoi(argv[3]); ss <= atoi(argv[4]); ss++)
//	{
//
//		delay = ss * dt;
//
//		Output_filename();
//
//
//		printf("Sample interval (Delay)=%0.3f\n", ss*dt);
//		Run_model();
//	}
//
//}


//int main(int argc, char **argv)          // scan time bin
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//	strcpy(input_filename, argv[1]);
//	double dt = atof(argv[2]);
//
//	for (int ss = atoi(argv[3]); ss <= atoi(argv[4]); ss++)
//	{
//
//		bin = ss * dt;
//
//		Output_filename();
//
//		printf("bin=%0.3f\n", ss*dt);
//		Run_model();
//	}
//
//}

//int main(int argc, char **argv)          // scan time bin 2^n
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//	strcpy(input_filename, argv[1]);
//
//	for (int ss = atoi(argv[2]); ss <= atoi(argv[3]); ss++)
//	{
//
//		bin = 2 * pow(0.5, ss);
//
//		Output_filename();
//		printf("bin=%0.3f\n", bin);
//		Run_model();
//	}
//
//}

//int main(int argc, char **argv)          // scan  S in HH,Lorenz,FN,Hn
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//	/*strcat(str, argv[1]);  ///model*/
//	p = atof(argv[2]);
//	s = atof(argv[3]);
//	f = atof(argv[4]);
//	u = atof(argv[5]);
//	double ds = atof(argv[6]);
//	double Strength;
//
//
//	for (int id = atoi(argv[7]); id <= atoi(argv[8]); id++)
//	{
//		Strength = int(id*ds * 1000) / 1000.0;
//
//		strcpy(str, argv[1]); ////model name
//		strcat(str, "p="), sprintf(c, "%0.2f", p), strcat(str, c);
//		strcat(str, "s="), sprintf(c, "%0.3f", Strength), strcat(str, c);
//		strcat(str, "f="), sprintf(c, "%0.3f", f), strcat(str, c);
//		strcat(str, "u="), sprintf(c, "%0.3f", u), strcat(str, c);
//		strcpy(input_filename, str);
//
//		Output_filename();
//		printf("s=%0.3f\n", Strength);
//		Run_model();
//	}
//
//
//}

//int main(int argc, char **argv)          // scan T_max
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//
//	Read_parameters();
//
//	strcpy(input_filename, argv[1]);
//	double dT = atof(argv[2]);
//	DT = dT;
//
//	for (int ss = atoi(argv[3]); ss <= atoi(argv[4]); ss++)
//	{
//		auto_T_max = 0;
//		T_Max = pow(2, ss) * dT;
//
//		Output_filename();
//
//		printf("T_max=%0.3e\n", T_Max);
//		Run_model();
//	}
//
//}



//int main(int argc, char **argv)          // scan  f&u in HH
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//
//	/*strcat(str, argv[1]);  ///model*/
//	p = atof(argv[2]);
//	s = atof(argv[3]);
//	f = atof(argv[4]);
//	u = atof(argv[5]);
//
//	double df, d_uf;
//	df = atof(argv[6]);
//	d_uf = atof(argv[7]);
//
//	for (int id_f = atoi(argv[8]); id_f <= atoi(argv[9]); id_f++)  
//	{
//		for (int id_uf = atoi(argv[10]); id_uf <= atoi(argv[11]); id_uf++) 
//		{
//
//			f = int(id_f * df * 1000) / 1000.0;
//			u = int((id_uf*d_uf) / f * 1000+0.5) / 1000.0;
//
//			//u = int((id_uf*d_uf)*0.05 / f * 1000) / 1000.0;  ///// for current based IF
//
//			strcpy(str, argv[1]); ////model name
//			strcat(str, "p="), sprintf(c, "%0.2f", p), strcat(str, c);
//			strcat(str, "s="), sprintf(c, "%0.3f", s), strcat(str, c);
//			strcat(str, "f="), sprintf(c, "%0.3f", f), strcat(str, c);
//			strcat(str, "u="), sprintf(c, "%0.3f", u), strcat(str, c);
//
//			strcpy(input_filename, str);
//
//			Output_filename();
//			printf("idf=%d id_uf=%d\n", id_f, id_uf);
//			Run_model();
//		}
//	}
//
//
//}


//int main(int argc, char **argv)          // scan  k in x_n^k  in HH,Lorenz,FN,Hn
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//	strcpy(input_filename, argv[1]);
//	double ds = atof(argv[2]);
//
//	for (int ss = atoi(argv[3]); ss <= atoi(argv[4]); ss++)
//	{
//		order[0] = ss;
//		Output_filename();
//		printf("k=%d\n", order[0]);
//		Run_model();
//	}
//}

//int main(int argc, char **argv)          // scan  l in y_n^l  in HH,Lorenz,FN,Hn
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//	strcpy(input_filename, argv[1]);
//	double ds = atof(argv[2]);
//
//	for (int ss = atoi(argv[3]); ss <= atoi(argv[4]); ss++)
//	{
//		order[1] = ss;
//		Output_filename();
//		printf("l=%d\n", order[1]);
//		Run_model();
//	}
//} 


//int main(int argc, char **argv)          // scan  S in IF
//{
//	double p, s, f, u;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//
//	if (argc > 2)
//	{
//		p = atof(argv[1]);
//		s = atof(argv[2]);
//		f = atof(argv[3]);
//		u = atof(argv[4]);
//	}
//
//	if (N == 3 || N == 5)
//	{
//		for (int ss = atoi(argv[5]); ss <= atoi(argv[6]); ss++)
//		{
//			strcpy(str, "p="), sprintf(c, "%0.3f", p), strcat(str, c);
//			strcat(str, "s="), sprintf(c, "%0.3f", ss*0.001), strcat(str, c);
//			strcat(str, "f="), sprintf(c, "%0.3f", f), strcat(str, c);
//			strcat(str, "u="), sprintf(c, "%0.3f", u), strcat(str, c);
//			strcpy(input_filename, str);
//
//			Output_filename();
//
//			printf("s=%0.3f\n", ss*0.001);
//			Run_model();
//		}
//	}
//	else
//	{
//		for (int IF = atoi(argv[5]); IF <= atoi(argv[6]); IF += 2)
//			for (int ir = atoi(argv[7]); ir <= atoi(argv[8]); ir++)
//			{
//				f = IF*0.001;
//				u = ir*0.1*0.05 / f;
//				u = int(u * 1000) / 1000.0;  // storage 3 digits
//				strcpy(str, "p="), sprintf(c, "%0.3f", p), strcat(str, c);
//				strcat(str, "s="), sprintf(c, "%0.3f", s), strcat(str, c);
//				strcat(str, "f="), sprintf(c, "%0.3f", f), strcat(str, c);
//				strcat(str, "u="), sprintf(c, "%0.3f", u), strcat(str, c);
//				strcpy(input_filename, str);
//
//				Output_filename();
//
//				printf("IF=%d ir=%d\n", IF, ir);
//				Run_model();
//			}
//	}
//
//}


//int main(int argc, char **argv)          // scan  T_max in IF
//{
//	double p, s, f, u, dt;
//	char str[200] = "", c[10];
//
//	Read_parameters();
//
//
//	if (argc > 2)
//	{
//		p = atof(argv[1]);
//		s = atof(argv[2]);
//		f = atof(argv[3]);
//		u = atof(argv[4]);
//		T_Max = atof(argv[5]);
//	}
//
//	if (N == 3 || N == 5)
//	{
//		for (int ss = atoi(argv[6]); ss <= atoi(argv[7]); ss++)
//		{		
//			strcpy(str, "p="), sprintf(c, "%0.3f", p), strcat(str, c);
//			strcat(str, "s="), sprintf(c, "%0.3f", s), strcat(str, c);
//			strcat(str, "f="), sprintf(c, "%0.3f", f), strcat(str, c);
//			strcat(str, "u="), sprintf(c, "%0.3f", u), strcat(str, c);
//			strcpy(input_filename, str);
//
//			Output_filename();
//
//			printf("Tmax=%0.3e\n", T_Max);
//			Run_model();
//			T_Max *= 2;
//		}
//	}
//	else
//	{
//		for (int IF = atoi(argv[5]); IF <= atoi(argv[6]); IF += 2)
//			for (int ir = atoi(argv[7]); ir <= atoi(argv[8]); ir++)
//			{
//				f = IF*0.001;
//				u = ir*0.1*0.05 / f;
//				u = int(u * 1000) / 1000.0;  // storage 3 digits
//				strcpy(str, "p="), sprintf(c, "%0.3f", p), strcat(str, c);
//				strcat(str, "s="), sprintf(c, "%0.3f", s), strcat(str, c);
//				strcat(str, "f="), sprintf(c, "%0.3f", f), strcat(str, c);
//				strcat(str, "u="), sprintf(c, "%0.3f", u), strcat(str, c);
//				strcpy(input_filename, str);
//
//				Output_filename();
//
//				printf("IF=%d ir=%d\n", IF, ir);
//				Run_model();
//			}
//	}
//
//}