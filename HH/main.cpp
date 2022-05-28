
/* Hodgkin-Huxley model: Network (Library and Regular)*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------


#include "common_header.h"
#include "Def.h"
#include "Gate_variables.h"
#include "Random.h"
#include "Read_parameters.h"
#include "Initialization.h"
#include "Find_cubic_hermite_root.h"
#include "Runge_Kutta4.h"
#include "Runge_Kutta2.h"
#include "ETD.h"
#include "ETDRK.h"
#include "test_func.h"
#include "Run_model.h"
#include "Largest_Lyapunov.h"
#include "Delete.h"
#include <fstream>

int main(int argc,char **argv) {
	long seed, seed0, seed1, seed2;
	clock_t t0, t1;	 
	char str[200];
	double MLE;
	double mean_fire_rate;
  bool verbose;
  // Config program options:
  po::options_description generic("Generic Options");
  generic.add_options()
    ("help,h", "produce help message")
    ("verbose,v", po::bool_switch(&verbose), "show output")
    ("config,c", po::value<string>()->default_value("NetModel_parameters.ini"), "config filename.")
    ;
  po::options_description config("Configs");
  config.add_options()
    ("NE",          po::value<int>()->default_value(2), "num of E neurons")
    ("NI",          po::value<int>()->default_value(0), "num of I neurons")
    ("seed",        po::value<string>()->default_value("11 11"), "seed to generate connectivity matrix and init Poisson generators.")
    ("T_Max",       po::value<double>()->default_value(1e7), "Simulation time period, unit ms.")
    ("T_step",      po::value<double>()->default_value(0.05), "Time step, unit ms.")
    ("S",           po::value<string>()->default_value("0.02 0.02 0.02 0.02"), "Synaptic coupling strength")
    ("I_CONST",     po::value<double>()->default_value(0), "Constant external drive.")
    ("Nu",          po::value<double>()->default_value(0.1), "Poisson input rate, unit kHz.")
    ("full_mode",   po::value<int>()->default_value(0), "Assign connectivity matrix and feed forward strength entry by entry.")
    ("f",           po::value<string>()->default_value(""), "FFWD Poisson strength, separated by space.")
    ("fE",          po::value<double>()->default_value(0.1), "Homogeneous FFWD Poisson strength for E neuron, if full_mode off.")
    ("fI",          po::value<double>()->default_value(0.1), "Homogeneous FFWD Poisson strength for I neuron, if full_mode off.")
    ("conn_matrix", po::value<string>()->default_value(""), "row-wise connectivity matrix, separated by space.")
    ("P_c",         po::value<double>()->default_value(0.25), "Erdos-Renyi connecting probability.")
    ("random_S",    po::value<int>()->default_value(0), "random mode of recurrent coupling strength (0-none 1-uniform 2-gauss 3-exponential 4-lognormal)")
    ("random_Nu",   po::value<int>()->default_value(0), "random mode of ffwd Poisson frequency (0-none 1-uniform 2-gauss 3-exponential 4-lognormal)")
    ("CP",          po::value<double>()->default_value(0), "Probability of correlated ffwd Poisson input.")
    ("Lyapunov",    po::value<int>()->default_value(0), "toggle to calculate Lyapunov exponent.")
    ("record_spk",  po::value<int>()->default_value(1), "toggle to record spike train.")
    ("record_v",    po::value<int>()->default_value(0), "toggle to record v.")
    ("record_vlim", po::value<string>()->default_value("0 1e8"), "time range to record voltage trace.")
    ("record_path", po::value<string>()->default_value("./data/"), "path to save data")
    ("state_path",  po::value<string>()->default_value(""), "path to load init state of neurons.")
    ;
  // create variable map
  po::variables_map vm;
  po::options_description cml_options;
  cml_options.add(generic).add(config);
  po::store(po::parse_command_line(argc, argv, cml_options), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << generic << '\n';
    cout << config << '\n';
    return 1;
  }
  // loading parsers from config file
  ifstream config_file;
  if (vm.count("config")) {
    string cfname = vm["config"].as<string>();
    config_file.open(cfname.c_str());
    po::store(po::parse_config_file(config_file, config), vm);
    po::notify(vm);
  }
  // Override config params with cml params
  po::store(po::parse_command_line(argc, argv, cml_options), vm);
  po::notify(vm);

  vector<int> seed_buff;
	str2vec(vm["seed"].as<string>(), seed_buff);
	seed = seed_buff[0]; 
	seed1 = seed_buff[1];

	Read_parameters(vm);
	out_put_filename();
	seed0 = seed;    // Create connect matrix
	seed2 = seed1;  // Initialization & Poisson
	Initialization(seed0, seed2);

	t0 = clock();
	if (Lyapunov)
		MLE = Largest_Lyapunov(seed2, 1, T_step);
	else
		Run_model();

	// save neuronal states
	SaveNeuronState();

	double total_fire_num[2] = { 0 };
	for (int i = 0; i < N; i++)
		total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;

	mean_fire_rate = (total_fire_num[0] + total_fire_num[1]) / T_Max * 1000 / N; //(Hz)
	printf("mean rate (Hz) = %0.2f ", mean_fire_rate);
	printf("(E : %.3f, I : %.3f)\n", total_fire_num[0] / T_Max * 1000 / NE, total_fire_num[1] / T_Max * 1000 / NI);

	t1 = clock();
	printf("Total time = %0.3fs \n\n", double(t1 - t0) / CLOCKS_PER_SEC);
	Delete();
}


//////////////////////// scan S
//int main(int argc,char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//
//	Read_parameters(seed, seed1);
//
//	printf("Run model name: %s\n", argv[1]); // model
//	P_c = atof(argv[2]);
//	S[0] = atof(argv[3]); S[1] = S[0]; S[2] = S[0]; S[3] = S[0];
//	f[0] = atof(argv[4]); f[1] = f[0];
//	Nu = atof(argv[5]);
//	double ds = atof(argv[6]);
//
//	/////////////////////
//	Lyapunov = 0;
//	record_data[0] = 1;
//	record_data[1] = 0;
//	Power_spectrum = 0;
//
//	for (int id = atoi(argv[7]); id <= atoi(argv[8]); id++)
//	{
//		S[0] = id*ds;
//
//		out_put_filename();
//		seed0 = seed;    // Create connect matrix
//		seed2 = seed1;  // Initialization & Poisson
//		Initialization(seed0, seed2);
//
//
//		t0 = clock();
//		Run_model();
//		t1 = clock();
//
//		int total_fire_num[2] = { 0 };
//		for (int i = 0; i < N; i++)
//			total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//		if (NE == N)
//			printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//		else if (NI == N)
//			printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//		else
//			printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//		
//		double rate = total_fire_num[0] / neu[0].t * 1000 / NE;
//
//		printf("s=%0.3f ",S[0]);
//		printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//		Delete();
//	}
//
//}

////////////////////////// scan f & u
//int main(int argc,char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//
//	Read_parameters(seed, seed1);
//
//	P_c = atof(argv[1]);
//	S[0] = atof(argv[2]); S[1] = S[0]; S[2] = S[0]; S[3] = S[0];
//	f[0] = atof(argv[3]); f[1] = f[0];
//	Nu = atof(argv[4]);
//
//	/////////////////////
//	Lyapunov = 0;
//	record_data[0] = 1;
//	record_data[1] = 0;
//	Power_spectrum = 0;
//	save_3_digital = 1;
//
//	for (int id_f = atoi(argv[5]); id_f <= atoi(argv[6]); id_f++)  ///0:1:15
//	{
//		for (int id_uf = atoi(argv[7]); id_uf <= atoi(argv[8]); id_uf++)  ///0:1:20
//		{
//			double df = 0.01, d_uf = 0.002;
//			f[0] = id_f * df+0.05, f[1] = f[0];
//			Nu = int((id_uf*d_uf + 0.01) / f[0] * 1000+0.5) / 1000.0;
//
//
//			out_put_filename();
//			seed0 = seed;    // Create connect matrix
//			seed2 = seed1;  // Initialization & Poisson
//			Initialization(seed0, seed2);
//
//
//			t0 = clock();
//			Run_model();
//			t1 = clock();
//
//			int total_fire_num[2] = { 0 };
//			for (int i = 0; i < N; i++)
//				total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//			if (NE == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//			else if (NI == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//			else
//				printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//
//			double rate = total_fire_num[0] / neu[0].t * 1000 / NE;
//
//			printf("s=%0.3f ", S[0]);
//			printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//			Delete();
//		}
//	}
//}


///// build library basic method
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200];
//
//	Read_parameters(seed, seed1);
//
//	NE = 1;
//	NI = 0;
//	N = 1;
//	T_Max = T_ref;
//	f[0] = 0;
//	S[0] = 0;
//	Nu = 0.0;
//	ode_type = 4;
//	Lib_method = 0;
//	Adaptive_method = 0;
//	Estimate_RK4_call = 0;
//	record_data[0] = 0;
//	record_data[1] = 0;
//	Power_spectrum = 0;
//	Lyapunov = 0;
//	T_Step_Large = pow(2, -10);
//	library_v_trace = 0;   ////
//	I_CONST = 1;     //////////////////// const way!
//
//	out_put_filename();
//	seed0 = seed;    // Create connect matrix
//	seed2 = seed1;  // Initialization & Poisson
//	Initialization(seed0, seed2);
//
//	if (I_CONST)
//	{
//		ffp = fopen("tttLib_const.dat", "wb");
//		//fp_v = fopen("tttLib_v_const.dat", "wb");
//		//fp_m = fopen("Lib_m_const.dat", "wb");
//		//fp_h = fopen("Lib_h_const.dat", "wb");
//		//fp_n = fopen("Lib_n_const.dat", "wb");
//	}
//	else
//	{
//		ffp = fopen("Lib_decay8.dat", "wb");
//		//fp_v = fopen("Lib_v_decay8.dat", "wb");
//		//fp_m = fopen("Lib_m_decay8.dat", "wb");
//		//fp_h = fopen("Lib_h_decay8.dat", "wb");
//		//fp_n = fopen("Lib_n_decay8.dat", "wb");	
//	}
//
//
//	for (int i = 0; i <= 20; i++) // I, 0:2.5:50
//		for (int j = 0; j <= 15; j++) // m, 0:0.02:0.3
//		{
//			t0 = clock();
//			for (int k = 0; k <= 20; k++) // h, 0.2:0.02:0.6
//				for (int l = 0; l <= 15; l++) //n, 0.3:0.02:0.6
//				{
//					neu[0].v = V_th;
//					neu[0].G_se = i*0.05;
//					neu[0].m = j*0.02/2;
//					neu[0].h = k*0.02/2 + 0.2;
//					neu[0].n = l*0.02/2 + 0.3;
//					neu[0].t = 0;
//					neu[0].dv = 0;
//					neu[0].G_f = 0;
//					neu[0].G_ff = 0;
//					neu[0].G_sse = 0;
//					neu[0].G_si = 0;
//					neu[0].G_ssi = 0;
//					neu[0].fire_num = 0;
//					neu[0].last_fire_time = 0;
//					neu[0].if_fired = 0;
//					double I = 2.5*i;
//
//					if (I_CONST)
//						I_const_input = I;
//
//					fwrite(&I, sizeof(double), 1, ffp);           // library
//					fwrite(&neu[0].m, sizeof(double), 1, ffp);
//					fwrite(&neu[0].h, sizeof(double), 1, ffp);
//					fwrite(&neu[0].n, sizeof(double), 1, ffp);
//	
//
//					//fwrite(&neu[0].v, sizeof(double), 1, fp_v);           // record v,m,h,n
//					//fwrite(&neu[0].m, sizeof(double), 1, fp_m);
//					//fwrite(&neu[0].h, sizeof(double), 1, fp_h);
//					//fwrite(&neu[0].n, sizeof(double), 1, fp_n);
//					//if(l==15)
//					//	printf("%d %d %d %d %f %f %f %f\n\n", i, j, k, l, I, neu[0].m, neu[0].h, neu[0].n);
//					Run_model();
//					fwrite(&neu[0].v, sizeof(double), 1, ffp);
//					fwrite(&neu[0].m, sizeof(double), 1, ffp);
//					fwrite(&neu[0].h, sizeof(double), 1, ffp);
//					fwrite(&neu[0].n, sizeof(double), 1, ffp);
//						
//				}
//			t1 = clock();
//			printf("i=%d j=%d time=%0.3fs\n",i,j, double(t1 - t0) / CLOCKS_PER_SEC);
//		}
//	fclose(ffp);
////	fclose(fp_v);// fclose(fp_m), fclose(fp_h), fclose(fp_n);
//	Delete();
//}

///////////////////Convergence tests
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//	FILE *FFP;
//
//
//	Read_parameters(seed, seed1);
//
//	ode_type = atoi(argv[1]);
//	S[0] = atof(argv[2]);
//	S[1] = S[0], S[2] = S[0], S[3] = S[0];
//	int ca = atoi(argv[3]);
//
//
//	//for (int ca = 1; ca <= 2; ca++) 	
//	//{ 
//		Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;
//		method = ca;
//		if (method <= 1)
//			Regular_method = 1;
//		else if (method == 2)
//			Lib_method = 1;
//		else if (method == 3)
//			ETDRK_method = 1;
//		else if (method == 4)
//			ETD_method = 1;
//		else
//		{
//			printf("Wrong method=%d\n", method);
//			exit(0);
//		}
//
//		strcpy(str, lib_name);
//		strcat(str, "_RK"), sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//
//		if (Lib_method)
//			strcat(str, "lib_");
//		else if (Adaptive_method)
//			strcat(str, "ad_");
//		else if(Regular_method)
//			strcat(str, "re_");
//		else if(ETDRK_method)
//			strcat(str, "ETDRK_");
//		else
//			strcat(str, "ETD_");
//
//		strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//		strcat(str, "s="), sprintf(c, "%0.2f", S[0]), strcat(str, c);
//		strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//		if (Nu < Epsilon)
//			strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//		else
//			strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//		strcat(str, "_convergence_tests.dat");
//
//		FFP = fopen(str, "wb");
//
//		for (int tt = 4; tt <= 13; tt++)          
//		{
//			if (tt == 13) 
//				tt = 13;
//			T_step = pow(0.5, tt);
//
//			out_put_filename();
//			seed0 = seed;    // Create connect matrix
//			seed2 = seed1;  // Initialization & Poisson
//			Initialization(seed0, seed2);
//
//			t0 = clock();
//			Run_model();
//
//			double total_fire_num[2] = { 0 };
//			for (int i = 0; i < N; i++)
//				total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//			if (NE == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//			else if (NI == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//			else
//				printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//			fwrite(&T_step, sizeof(double), 1, FFP);
//			for (int i = 0; i < N; i++)
//			{
//				fwrite(&neu[i].v, sizeof(double), 1, FFP);
//				fwrite(&neu[i].m, sizeof(double), 1, FFP);
//				fwrite(&neu[i].h, sizeof(double), 1, FFP);
//				fwrite(&neu[i].n, sizeof(double), 1, FFP);
//				fwrite(&neu[i].last_fire_time, sizeof(double), 1, FFP);
//			}
//
//			for (int i = 0; i < N; i++)
//			{
//				printf("%d %f\n",i,neu[i].last_fire_time);
//			}
//
//			t1 = clock();
//			printf("tt=%d s=%0.3f ", tt, S[0]);
//			printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//			Delete();
//		}
//		fclose(FFP);
//	//}
//}

///////// MLE
//int main(int argc,char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//
//
//	Read_parameters(seed, seed1);
//
//	T_step= atof(argv[1]);
//	method = atoi(argv[2]);
//
//	Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;
//	if (method <= 1)
//		Regular_method = 1;
//	else if (method == 2)
//		Lib_method = 1;
//	else if (method == 3)
//		ETDRK_method = 1;
//	else if (method == 4)
//		ETD_method = 1;
//	else
//	{
//		printf("Error! method=%d\n", method);
//		exit(0);
//	}
//
//	/////////////////////
//	Lyapunov = 1;
//	record_data[0] = 0;
//	record_data[1] = 0;
//	Power_spectrum = 0;
//	Estimate_RK4_call = 0;
//	T_Max = 6e4;
//
//	strcpy(str, "MLE_"), strcat(str, lib_name), strcat(str, "_");
//	
//	if (Lib_method)
//	{
//		strcat(str, "RK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//		strcat(str, "lib_");
//	}
//	else if (ETD_method)
//	{
//		strcat(str, "ETD");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//	else if (ETDRK_method)
//	{
//		strcat(str, "ETDRK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//	else
//	{
//		strcat(str, "RK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//
//	double tt = atof(argv[1]);
//	if (fabs(tt - int(tt)) < 1e-8)
//	{
//		strcat(str, "t="), sprintf(c, "%0.1f", tt), strcat(str, c);
//	}
//	else
//	{
//		strcat(str, "t="), sprintf(c, "%0.3f", tt), strcat(str, c);
//	}
//
//	strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//	strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//	if (Nu < Epsilon)
//		strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//	else
//		strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//	strcat(str, ".dat");
//	ffp = fopen(str, "wb");
//
//	double *SS;
//	int length_SS;
//
//	if (Nu < Epsilon)
//	{
//		SS = new double[41];
//		for (int i = 0; i < 41; i++)
//			SS[i] = 0.4 + i*0.01;
//		length_SS = 41;
//	}
//	else
//	{
//		if (P_c == 1 || P_c == 0.15)
//		{
//			SS = new double[41];
//			for (int i = 0; i < 41; i++)
//				SS[i] = i*(P_c == 0.15 ? 0.2 : 0.05);
//			length_SS = 41;
//		}
//		else if (P_c == 0.1)
//		{
//			SS = new double[51];
//			for (int i = 0; i < 51; i++)
//				SS[i] = i* 0.2;
//			length_SS = 51;
//		}
//		else
//		{
//			printf("Ensure Pc = %0.3f\n",P_c);
//			exit(0);
//		}
//
//	}
//	
//	for (int id = 0; id < length_SS; id++)
//	{
//		S[0] = SS[id];
//
//		out_put_filename();
//		seed0 = seed;    // Create connect matrix
//		seed2 = seed1;  // Initialization & Poisson
//		Initialization(seed0, seed2);
//
//
//		t0 = clock();
//
//		MLE = Largest_Lyapunov(seed2, 1, T_step);
//
//		double total_fire_num[2] = { 0 };
//		for (int i = 0; i < N; i++)
//			total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//		if (NE == N)
//			printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//		else if (NI == N)
//			printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//		else
//			printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//		t1 = clock();
//
//		double rate = total_fire_num[0] / neu[0].t * 1000 / NE;
//
//		fwrite(&S[0], sizeof(double), 1, ffp);
//		fwrite(&rate, sizeof(double), 1, ffp);
//		fwrite(&MLE, sizeof(double), 1, ffp);
//
//		printf("s=%0.3f ",S[0]);
//		printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//		Delete();
//	}
//	fclose(ffp);
//}


//////hopf bifurcation 
//int main(int argc, char **argv) 
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200],c[10];
//	double MLE;
//	double mean_fire_rate;
//
//
//	Read_parameters(seed, seed1);
//
//	T_step = atof(argv[1]);
//	method = atoi(argv[2]);
//	int choose = atoi(argv[3]);  /// 1--hopf1, 2--hopf2
//	int End[2] = { 60,40 };
////	NE = atoi(argv[4]);
//	
//	if (choose == 1)
//		T_Max = 1e4;
//	else if(choose == 2)
//		T_Max = 200;
//	else
//	{
//		printf("Wrong Hopf %d\n", choose);
//		exit(0);
//	}
//
//	if (method <= 1)
//		Regular_method = 1;
//	else if (method == 2)
//		Lib_method = 1;
//	else if (method == 3)
//		ETDRK_method = 1;
//	else if (method == 4)
//		ETD_method = 1;
//	else
//	{
//		printf("Error! method=%d\n", method);
//		exit(0);
//	}
//	
//
//	/////////////////////
//	I_CONST = 1;
//	NE = 1;		////
//	NI = 0;
//	N = NE+NI;
//	Lyapunov = 0;
//	Power_spectrum = 0;
//	Estimate_RK4_call = 0;
//	record_data[0] = 0, record_data[1] = 0;
//
//	if (T_Max == 200)
//		strcpy(str, "HB2_");
//	else
//		strcpy(str, "HB1_");
//	
//	strcat(str, lib_name), strcat(str, "_");			
//
//	if (Lib_method)
//	{
//		strcat(str, "RK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//		strcat(str, "lib_");
//	}
//	else if (ETD_method)
//	{
//		strcat(str, "ETD");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//	else if (ETDRK_method)
//	{
//		strcat(str, "ETDRK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//	else
//	{
//		strcat(str, "RK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//
//	double tt = atof(argv[1]);
//	if (fabs(tt - int(tt)) < 1e-8)
//	{
//		strcat(str, "t="), sprintf(c, "%0.1f", tt), strcat(str, c);
//	}
//	else
//	{
//		strcat(str, "t="), sprintf(c, "%0.3f", tt), strcat(str, c);
//	}
//
//	if (N > 1)
//	{
//		strcat(str, "n="), sprintf(c, "%d", N), strcat(str,c);
//		strcat(str, "s="), sprintf(c, "%0.2f", S[0]), strcat(str, c);
//	}
//	strcat(str,".dat");
//
//
//
//	ffp = fopen(str,"wb");
//	for (int ii = 0; ii <= End[choose-1]; ii++) //a   60 & 40 , 1e4, 200
//	{
//		if(choose == 1)
//			I_const_input = 5 + ii*0.05;
//		else
//			I_const_input = 5.9 + ii*0.01;
//		
//
//		fwrite(&I_const_input,sizeof(double),1,ffp);
//
//		out_put_filename();
//		seed0 = seed;    // create connect matrix
//		seed2 = seed1;  // initialization & poisson
//		Initialization(seed0, seed2);
//
//		t0 = clock();
//		Run_model();
//
//		double total_fire_num[2] = { 0 };
//		for (int i = 0; i < N; i++)
//			total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//		if (NE == N)
//			printf("mean rate(hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//		else if (NI == N)
//			printf("mean rate(hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//		else
//			printf("mean rate(hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//		double rate = total_fire_num[0] / neu[0].t * 1000 / NE;
//		fwrite(&rate, sizeof(double), 1, ffp);
//
//		t1 = clock();
//		printf("total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//		Delete();
//	}
//	fclose(ffp);
//}


//////
////library error using for once__IntOrd-th interpolation,1--linear,2--quadratic 
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//
//	Read_parameters(seed, seed1);
//	int interpolation_order = atoi(argv[1]); //////			
//
//	NE = 1;
//	NI = 0;
//	N = NE + NI;
//	T_Max = T_ref;
//	f[0] = 0;
//	S[0] = 0;
//	P_c = 0;
//	Nu = 0.0;
//	ode_type = 4;
//	Lib_method = 0;
//	Adaptive_method = 0;
//	record_data[0] = 0;
//	record_data[1] = 0;
//	Power_spectrum = 0;
//	Estimate_RK4_call = 0;
//	Lyapunov = 0;
//	T_Step_Large = pow(2, -10);
//	library_v_trace = 0;
//	I_CONST = 1;	   //////////////////// const way!
//
//	out_put_filename();
//	seed0 = seed;    // Create connect matrix
//	seed2 = seed1;  // Initialization & Poisson
//	Initialization(seed0, seed2);
//
//	double dG, dm, dh, dn;
//	dG = 0.05, dm = 0.02, dh = 0.02, dn = 0.02;
//
//	double X0[4], X[4];  //output v,m,h,n, X0=true value, X=approximation
//
//	strcpy(str, lib_name), strcat(str, "_LibError_");    //////////////
//	sprintf(c, "%d", interpolation_order), strcat(str, c), strcat(str, "_once.dat");
//	ffp = fopen(str, "wb");
//
//	neu[0].v = V_th;
//	neu[0].G_se = 0.16;			// I_input = 8
//	neu[0].m = 0.12;
//	neu[0].h = 0.48;
//	neu[0].n = 0.37;
//
//	neu[0].t = 0;
//	neu[0].dv = 0;
//	neu[0].G_f = 0;
//	neu[0].G_ff = 0;
//	neu[0].G_sse = 0;
//	neu[0].G_si = 0;
//	neu[0].G_ssi = 0;
//	neu[0].fire_num = 0;
//	neu[0].last_fire_time = 0;
//	neu[0].if_fired = 0;
//
//	I_const_input = -neu[0].G_se*(V_th - V_G_E);
//
//	Run_model();
//
//	X0[0] = neu[0].v, X0[1] = neu[0].m;
//	X0[2] = neu[0].h, X0[3] = neu[0].n;
//
//	for (int tt = -1; tt <= 10; tt++)
//	{
//		double c = pow(0.5, tt);
//		for (int j = 0; j < 4; j++)
//			X[j] = 0;
//
//		t0 = clock();
//
//		for (int i = 0; i < int(pow((interpolation_order + 1), 4)+0.01); i++)  //4 parameters, I,m,h,n
//		{
//			int a[4];
//			int k = i;
//			double s = 1;
//			for (int j = 0; j < 4; j++)
//			{
//				a[j] = k % (interpolation_order + 1);
//				k /= (interpolation_order + 1);
//				for (int ll = 0; ll < interpolation_order + 1; ll++)
//				{
//					if (ll == a[j])
//						continue;
//					else
//						s *= (0.5 - ll) / (a[j] - ll);  ///// input = x_0+dx/2
//				}
//			}
//
//			neu[0].v = V_th;
//			neu[0].G_se = 0.16 + (a[3] - 0.5)*dG*c; ////// set I,m,h,n, I=I_0+dI/2,{I_0,I_1,I_2,...,I_interpolation_order}
//			neu[0].m = 0.12 + (a[2] - 0.5)*dm*c;
//			neu[0].h = 0.48 + (a[1] - 0.5)*dh*c;
//			neu[0].n = 0.37 + (a[0] - 0.5)*dn*c;
//
//			neu[0].t = 0;
//			neu[0].dv = 0;
//			neu[0].G_f = 0;
//			neu[0].G_ff = 0;
//			neu[0].G_sse = 0;
//			neu[0].G_si = 0;
//			neu[0].G_ssi = 0;
//			neu[0].fire_num = 0;
//			neu[0].last_fire_time = 0;
//			neu[0].if_fired = 0;
//
//			I_const_input = -neu[0].G_se*(V_th - V_G_E);
//
//			Run_model();
//
//			X[0] += neu[0].v * s, X[1] += neu[0].m * s;
//			X[2] += neu[0].h * s, X[3] += neu[0].n * s;
//		}
//
//
//		t1 = clock();
//
//		fwrite(&c, sizeof(double), 1, ffp);
//		double sum_error[4];
//		for (int j = 0; j < 4; j++)
//			sum_error[j] = abs(X0[j] - X[j]);
//													
//		fwrite(&sum_error, sizeof(double), 4, ffp);
//		printf("tt=%d time=%0.3fs\n", tt, double(t1 - t0) / CLOCKS_PER_SEC);
//	}
//	fclose(ffp);
//	Delete();
//
//}

///////// library error vs time
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//
//
//	Read_parameters(seed, seed1);
//	if (argc > 5)
//	{
//		T_Step_Large = pow(0.5, atof(argv[1])), T_Step_Small = pow(0.5, atof(argv[2]));
//		Lib_method = atoi(argv[3]), Adaptive_method = atoi(argv[4]);
//		//S[0] = atof(argv[5]); S[1] = S[0]; S[2] = S[0]; S[3] = S[0];
//		IntOrd = atoi(argv[5]);
//	}
//	I_const_input = 8;
//
//	strcpy(str, file), strcat(str, lib_name), strcat(str, "_LibError_");
//	sprintf(c, "%d", IntOrd), strcat(str, c), strcat(str, "_RK");   /////////
//	sprintf(c,"%d",ode_type), strcat(str, c), strcat(str, "_");
//		
//	double n, m;
//	n = log(T_Step_Large) / log(0.5);
//	m = log(T_Step_Small) / log(0.5);
//	if (Lib_method == 0)
//	{
//		strcat(str, "t_l="), sprintf(c, "%0.1f", n), strcat(str, c);
//		strcat(str, "t_s="), sprintf(c, "%0.1f", m), strcat(str, c);
//	}
//	else
//	{
//		strcat(str, "lib_t="), sprintf(c, "%0.1f", n), strcat(str, c);
//	}
//	
//		strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//		strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//		if (Nu < Epsilon)
//			strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//		else
//			strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//	
//		strcat(str, ".dat");
//		ffp = fopen(str, "wb");
//
//	/////////////////////	
//	out_put_filename();
//	seed0 = seed;    // Create connect matrix
//	seed2 = seed1;  // Initialization & Poisson
//	Initialization(seed0, seed2);
//
//	t0 = clock();
//	if (Lyapunov)
//		MLE = Largest_Lyapunov(seed2, 1, T_Step_Large);
//	else
//		Run_model();
//
//
//
//	double total_fire_num[2] = { 0 };
//	for (int i = 0; i < N; i++)
//		total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//
//	mean_fire_rate = (total_fire_num[0] + total_fire_num[1]) / T_Max * 1000 / N; //(Hz)
//	printf("mean rate (Hz) = %f ", mean_fire_rate);
//
//	t1 = clock();
//
//	printf("Total time = %0.3fs \n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//	Delete();
//	fclose(ffp);
//
//}

////////// Efficiency  output: S, rate, p, p1, num1, num2, time
//int main(int argc,char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//
//	Read_parameters(seed, seed1);
//
//	T_step= atof(argv[1]);
//	method = atof(argv[2]);
//
//	Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;
//	if (method == 0)
//		Regular_method = 1;
//	else if (method == 1)
//		Adaptive_method = 1;
//	else if (method == 2)
//		Lib_method = 1;
//	else if (method == 3)
//		ETDRK_method = 1;
//	else if (method == 4)
//		ETD_method = 1;
//	else
//	{
//		printf("Error! method=%d\n", method);
//		exit(0);
//	}
//
//	/////////////////////
//	Lyapunov = 0;
//	Power_spectrum = 0;
//	record_data[0] = 0;
//	record_data[1] = 0;
//	Estimate_RK4_call = 1;
//	T_Max = 1e4;   //edit
//	ode_type = 2;
//
//	strcpy(str, "EFFI_"), strcat(str, lib_name), strcat(str, "_");
//
//
//	if (Lib_method)
//	{
//		strcat(str, "RK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//		strcat(str, "lib_");
//	}
//	else if (ETD_method)
//	{
//		strcat(str, "ETD");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//	else if (ETDRK_method)
//	{
//		strcat(str, "ETDRK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//	else
//	{
//		strcat(str, "RK");
//		sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//	}
//	double tt = atof(argv[1]);
//	if (fabs(tt - int(tt)) < 1e-8)
//	{
//		strcat(str, "t="), sprintf(c, "%0.1f", tt), strcat(str, c);
//	}
//	else
//	{
//		strcat(str, "t="), sprintf(c, "%0.3f", tt), strcat(str, c);
//	}
//
//	strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//	strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//	if (Nu < Epsilon)
//		strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//	else
//		strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//	strcat(str, ".dat");
//	ffp = fopen(str, "wb");
//
//
//	double *SS;
//	int length_SS;
//
//	if (Nu < Epsilon)
//	{
//		SS = new double[41];
//		for (int i = 0; i < 41; i++)
//			SS[i] = 0.4 + i*0.01;
//		length_SS = 41;
//	}
//	else
//	{
//		if (P_c == 1 || P_c == 0.15)
//		{
//			SS = new double[41];
//			for (int i = 0; i < 41; i++)
//				SS[i] = i*(P_c == 0.15 ? 0.2 : 0.05);
//			length_SS = 41;
//		}
//		else if (P_c == 0.1)
//		{
//			SS = new double[51];
//			for (int i = 0; i < 51; i++)
//				SS[i] = i* 0.2;
//			length_SS = 51;
//		}
//		else
//		{
//			printf("Ensure Pc = %0.3f\n", P_c);
//			exit(0);
//		}
//	}
//
//	for (int id = 0; id < length_SS; id++)
//	{
//		S[0] = SS[id];
//		Call_num = 0;
//		syn_num = 0;
//		count_num = 0;
//
//		out_put_filename();
//		seed0 = seed;    // Create connect matrix
//		seed2 = seed1;  // Initialization & Poisson
//		Initialization(seed0, seed2);
//
//
//		t0 = clock();
//		Run_model();
//		t1 = clock();
//
//		double total_fire_num[2] = { 0 };
//		for (int i = 0; i < N; i++)
//			total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//		mean_fire_rate = (total_fire_num[0] + total_fire_num[1]) / T_Max * 1000 / N;
//		double s = 0;
//		printf("s=%0.3f mean rate(Hz) = %0.3f num1=%0.0f num2=%0.0f %f\n", \
//			S[0], mean_fire_rate, Call_num, s, s / Call_num);
//
//
//		fwrite(&S[0], sizeof(double), 1, ffp);
//		fwrite(&mean_fire_rate, sizeof(double), 1, ffp);
//
//		double p, p1;
//		p = mean_fire_rate*T_ref / 1000;
//		p1 = count_num == 0 ? 0 : syn_num / count_num;
//		fwrite(&p, sizeof(double), 1, ffp);
//		fwrite(&p1, sizeof(double), 1, ffp);
//		fwrite(&Call_num, sizeof(double), 1, ffp);
//		fwrite(&s, sizeof(double), 1, ffp);
//
//		s = double(t1 - t0) / CLOCKS_PER_SEC;
//		fwrite(&s, sizeof(double), 1, ffp);
//
//		printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//		Delete();		
//	}
//
//	fclose(ffp);
//}


/////////////////effi vs dt
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200]="", c[10];
//	double MLE;
//	double mean_fire_rate;
//	FILE *FFP;
//
//
//	Read_parameters(seed, seed1);
//
//	ode_type = atoi(argv[1]);
//	S[0] = atof(argv[2]);
//	S[1] = S[0], S[2] = S[0], S[3] = S[0];
//	int ca = atoi(argv[3]);
//
//	//for (int ca = 1; ca <= 2; ca++) 	
//	//{ 
//		Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;
//		method = ca;
//		if (method <= 1)
//			Regular_method = 1;
//		else if (method == 2)
//			Lib_method = 1;
//		else if (method == 3)
//			ETDRK_method = 1;
//		else if (method == 4)
//			ETD_method = 1;
//		else
//		{
//			printf("Wrong method=%d\n", method);
//			exit(0);
//		}
//
//		if (Lib_method)
//		{
//			strcat(str, "RK");
//			sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//			strcat(str, "lib_");
//		}
//		else if (ETD_method)
//		{
//			strcat(str, "ETD");
//			sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//		}
//		else if (ETDRK_method)
//		{
//			strcat(str, "ETDRK");
//			sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//		}
//		else
//		{
//			strcat(str, "RK");
//			sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//		}
//
//		strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//		strcat(str, "s="), sprintf(c, "%0.2f", S[0]), strcat(str, c);
//		strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//		if (Nu < Epsilon)
//			strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//		else
//			strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//		strcat(str, "_Effi.dat");
//
//		FFP = fopen(str, "wb");
//
//		int END_tt;
//		if (Regular_method)
//			END_tt = 0;
//		else
//			END_tt = 14;
//
//
//		for (int tt = 0; tt <= END_tt; tt++)           
//		{
//			T_step = 0.01+ 0.02*tt;
//			if (tt == 14)
//				T_step = 0.277;
//
//			out_put_filename();
//			seed0 = seed;    // Create connect matrix
//			seed2 = seed1;  // Initialization & Poisson
//			Initialization(seed0, seed2);
//
//			t0 = clock();
//			Run_model();
//
//			double total_fire_num[2] = { 0 };
//			for (int i = 0; i < N; i++)
//				total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//			if (NE == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//			else if (NI == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//			else
//				printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//
//			t1 = clock();
//			printf("tt=%d s=%0.3f ", tt, S[0]);
//			printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//
//			double ss = double(t1 - t0) / CLOCKS_PER_SEC;
//			fwrite(&T_step, sizeof(double), 1, FFP);
//			fwrite(&ss, sizeof(double), 1, FFP);
//
//			Delete();
//		}
//		fclose(FFP);
//	//}
//}

/////////effi vs S
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//	FILE *FFP;
//
//
//	Read_parameters(seed, seed1);
//
//	T_step = atof(argv[1]);
//	int ca = atoi(argv[2]);
//
//	//for (int ca = 1; ca <= 2; ca++) 	
//	//{ 
//		Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;
//		method = ca;
//		if (method <= 1)
//			Regular_method = 1;
//		else if (method == 2)
//			Lib_method = 1;
//		else if (method == 3)
//			ETDRK_method = 1;
//		else if (method == 4)
//			ETD_method = 1;
//		else
//		{
//			printf("Wrong method=%d\n", method);
//			exit(0);
//		}
//
//		strcpy(str, lib_name);
//		strcat(str, "_RK"), sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//
//		if (Lib_method)
//			strcat(str, "lib_");
//		else if (Adaptive_method)
//			strcat(str, "ad_");
//		else if(Regular_method)
//			strcat(str, "re_");
//		else if(ETDRK_method)
//			strcat(str, "ETDRK_");
//		else
//			strcat(str, "ETD_");
//
//		double s = log(T_step) / log(0.5);
//		if (fabs(s - int(s)) < 1e-8)
//		{
//			strcat(str, "t="), sprintf(c, "%0.1f", s), strcat(str, c);
//		}
//		else
//		{
//			strcat(str, "t="), sprintf(c, "%0.3f", T_step), strcat(str, c);
//		}
//
//		strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//		strcat(str, "s="), sprintf(c, "%0.2f", S[0]), strcat(str, c);
//		strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//		if (Nu < Epsilon)
//			strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//		else
//			strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//		strcat(str, "_Effi.dat");
//
//		FFP = fopen(str, "wb");
//		for (int ds = 0; ds <= 15; ds++)           
//		{
//			S[0] = ds * 0.1;
//			S[1] = S[0], S[2] = S[0], S[3] = S[0];
//
//			out_put_filename();
//			seed0 = seed;    // Create connect matrix
//			seed2 = seed1;  // Initialization & Poisson
//			Initialization(seed0, seed2);
//
//			t0 = clock();
//			Run_model();
//			t1 = clock();
//
//			double total_fire_num[2] = { 0 };
//			for (int i = 0; i < N; i++)
//				total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//			if (NE == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//			else if (NI == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//			else
//				printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//			double ss = double(t1 - t0) / CLOCKS_PER_SEC;
//			fwrite(&S[0], sizeof(double), 1, FFP);
//			fwrite(&ss, sizeof(double), 1, FFP);
//	
//			printf("ids=%d s=%0.3f ", ds, S[0]);
//			printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//			Delete();
//		}
//		fclose(FFP);
//	//}
//}

/////////rates vs S
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//	FILE *FFP;
//
//
//	Read_parameters(seed, seed1);
//
//	T_step = atof(argv[1]);
//	int ca = atoi(argv[2]);
//
//	//for (int ca = 1; ca <= 2; ca++) 	
//	//{ 
//		Regular_method = 0, Lib_method = 0, Adaptive_method = 0, ETDRK_method = 0, ETD_method = 0;
//		method = ca;
//		if (method <= 1)
//			Regular_method = 1;
//		else if (method == 2)
//			Lib_method = 1;
//		else if (method == 3)
//			ETDRK_method = 1;
//		else if (method == 4)
//			ETD_method = 1;
//		else
//		{
//			printf("Wrong method=%d\n", method);
//			exit(0);
//		}
//
//		strcpy(str, lib_name);
//		strcat(str, "_RK"), sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//
//		if (Lib_method)
//			strcat(str, "lib_");
//		else if (Adaptive_method)
//			strcat(str, "ad_");
//		else if(Regular_method)
//			strcat(str, "re_");
//		else if(ETDRK_method)
//			strcat(str, "ETDRK_");
//		else
//			strcat(str, "ETD_");
//
//		double s = log(T_step) / log(0.5);
//		if (fabs(s - int(s)) < 1e-8)
//		{
//			strcat(str, "t="), sprintf(c, "%0.1f", s), strcat(str, c);
//		}
//		else
//		{
//			strcat(str, "t="), sprintf(c, "%0.3f", T_step), strcat(str, c);
//		}
//
//		strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//		strcat(str, "s="), sprintf(c, "%0.2f", S[0]), strcat(str, c);
//		strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//		if (Nu < Epsilon)
//			strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//		else
//			strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//		strcat(str, "_rates.dat");
//
//		FFP = fopen(str, "wb");
//		for (int ds = 0; ds <= 15; ds++)           
//		{
//			S[0] = ds * 0.1;
//			S[1] = S[0], S[2] = S[0], S[3] = S[0];
//
//			out_put_filename();
//			seed0 = seed;    // Create connect matrix
//			seed2 = seed1;  // Initialization & Poisson
//			Initialization(seed0, seed2);
//
//			t0 = clock();
//			Run_model();
//
//			double total_fire_num[2] = { 0 };
//			for (int i = 0; i < N; i++)
//				total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//			if (NE == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//			else if (NI == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//			else
//				printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//			double ss = (total_fire_num[0] + total_fire_num[1]) / neu[0].t / N * 1000;
//			fwrite(&S[0], sizeof(double), 1, FFP);
//			fwrite(&ss, sizeof(double), 1, FFP);
//
//
//			t1 = clock();
//			printf("ids=%d s=%0.3f ", ds, S[0]);
//			printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//			Delete();
//		}
//		fclose(FFP);
//	//}
//}

/////////////////////////Convergence tests for firing rate
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200], c[10];
//	double MLE;
//	double mean_fire_rate;
//
//
//	Read_parameters(seed, seed1);
//
//
//	Lib_method = atoi(argv[1]), Adaptive_method = atoi(argv[2]);
//	S[0] = atof(argv[3]); S[1] = S[0]; S[2] = S[0]; S[3] = S[0];
//	
//
//	//for (int ca = 1; ca <= 3; ca++)
//	//{
//	//	if (ca == 1)
//	//	{
//	//		Lib_method = 0;
//	//		Adaptive_method = 0;
//	//	}
//	//	else if (ca == 2)
//	//	{
//	//		Lib_method = 0;
//	//		Adaptive_method = 1;
//	//	}
//	//	else
//	//	{
//	//		Lib_method = 1;
//	//		Adaptive_method = 0;
//	//	}
//
//
//		strcpy(str, lib_name);
//		strcat(str, "_RK"), sprintf(c, "%d", ode_type), strcat(str, c), strcat(str, "_");
//
//
//		if (Lib_method)
//			strcat(str, "lib_");
//		else if (Adaptive_method)
//			strcat(str, "ad_");
//		else
//			strcat(str, "re_");
//
//		strcat(str, "p="), sprintf(c, "%0.2f", P_c), strcat(str, c);
//		strcat(str, "s="), sprintf(c, "%0.2f", S[0]), strcat(str, c);
//		strcat(str, "f="), sprintf(c, "%0.2f", f[0]), strcat(str, c);
//		if (Nu < Epsilon)
//			strcat(str, "w="), sprintf(c, "%0.2f", Omega), strcat(str, c);
//		else
//			strcat(str, "u="), sprintf(c, "%0.2f", Nu), strcat(str, c);
//
//		strcat(str, "_convergence_tests_rate.dat");
//
//		ffp = fopen(str, "wb");
//		for (int tt = 4; tt <= 10; tt++)
//		{
//			//if (tt == 10)
//			//	tt = 12;
//			T_Step_Large = pow(0.5, tt);
//			if (Adaptive_method)
//				T_Step_Small = pow(0.5, 10);
//			else
//				T_Step_Small = T_Step_Large;
//
//			out_put_filename();
//			seed0 = seed;    // Create connect matrix
//			seed2 = seed1;  // Initialization & Poisson
//			Initialization(seed0, seed2);
//
//
//			t0 = clock();
//
//			Run_model();
//
//			double total_fire_num[2] = { 0 };
//			for (int i = 0; i < N; i++)
//				total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;
//
//			if (NE == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE);
//			else if (NI == N)
//				printf("mean rate(Hz) = %0.3f\n", total_fire_num[1] / neu[0].t * 1000 / NI);
//			else
//				printf("mean rate(Hz) = %0.3f %0.3f\n", total_fire_num[0] / neu[0].t * 1000 / NE, total_fire_num[1] / neu[0].t * 1000 / NI);
//
//			double s = total_fire_num[0] / neu[0].t * 1000 / NE;
//			//fprintf(ffp, "%f ", T_Step_Large);
//			fwrite(&T_Step_Large, sizeof(double), 1, ffp);
//			fwrite(&s, sizeof(double), 1, ffp);		
//
//			t1 = clock();
//			printf("tt=%d s=%0.3f ", tt, S[0]);
//			printf("Total time = %0.3fs\n\n", double(t1 - t0) / CLOCKS_PER_SEC);
//			Delete();
//		}
//		fclose(ffp);
////	}
//}

/////// library error using for once__only linear
//int main(int argc, char **argv)
//{
//	long seed, seed0, seed1, seed2;
//	clock_t t0, t1;
//	char str[200];
//
//	Read_parameters(seed, seed1);
//
//	NE = 1;
//	NI = 0;
//	N = NE+NI;
//	T_Max = T_ref;
//	f[0] = 0;
//	S[0] = 0;
//	P_c = 0;
//	Nu = 0.0;
//	ode_type = 4;
//	Lib_method = 0;
//	Adaptive_method = 0;
//	record_data[0] = 0;
//	record_data[1] = 0;
//	Power_spectrum = 0;
//	Estimate_RK4_call = 0;
//	Lyapunov = 0;
//	T_Step_Large = pow(2, -16);
//	library_v_trace = 0;
//	I_CONST = 1;	   //////////////////// const way!
//
//	out_put_filename();
//	seed0 = seed;    // Create connect matrix
//	seed2 = seed1;  // Initialization & Poisson
//	Initialization(seed0, seed2);
//
//	double dG, dm, dh, dn;
//	dG = 0.05, dm = 0.02, dh = 0.02, dn = 0.02;
//
//	int id[4];  // G,m,h,n
//	double X0[4], X[4];  //output v,m,h,n (mean value)
//
//	strcpy(str, lib_name), strcat(str, "_LibError_linear_once.dat");    //////////////
//	ffp = fopen(str, "wb");
//
//	neu[0].v = V_th;
//	neu[0].G_se = 0.16;			// I_input = 8
//	neu[0].m = 0.12;
//	neu[0].h = 0.48;
//	neu[0].n = 0.37;
//
//	neu[0].t = 0;
//	neu[0].dv = 0;
//	neu[0].G_f = 0;
//	neu[0].G_ff = 0;
//	neu[0].G_sse = 0;
//	neu[0].G_si = 0;
//	neu[0].G_ssi = 0;
//	neu[0].fire_num = 0;
//	neu[0].last_fire_time = 0;
//	neu[0].if_fired = 0;
//
//	I_const_input = -neu[0].G_se*(V_th - V_G_E);
//
//	Run_model();
//
//	X0[0] = neu[0].v;
//	X0[1] = neu[0].m;
//	X0[2] = neu[0].h;
//	X0[3] = neu[0].n;
//
//	for (int tt = -1; tt <= 10; tt++)
//	{
//		double s = pow(0.5, tt);
//		for (int j = 0; j < 4; j++)
//			X[j] = 0;
//
//		t0 = clock();
//		for (int i = 0; i < 16; i++)
//		{
//			int k = i;
//			for (int j = 0; j < 4; j++)
//			{
//				id[3 - j] = k % 2;
//				k /= 2;
//			}
//
//			neu[0].v = V_th;
//			neu[0].G_se = 0.16 + (2 * id[0] - 1)*dG*s/2;			// I_input = 8
//			neu[0].m = 0.12 + (2 * id[1] - 1)*dm*s/2;
//			neu[0].h = 0.48 + (2 * id[2] - 1)*dh*s/2;
//			neu[0].n = 0.37 + (2 * id[3] - 1)*dn*s/2;
//
//
//			neu[0].t = 0;
//			neu[0].dv = 0;
//			neu[0].G_f = 0;
//			neu[0].G_ff = 0;
//			neu[0].G_sse = 0;
//			neu[0].G_si = 0;
//			neu[0].G_ssi = 0;
//			neu[0].fire_num = 0;
//			neu[0].last_fire_time = 0;
//			neu[0].if_fired = 0;
//
//			I_const_input = -neu[0].G_se*(V_th - V_G_E);
//
//			Run_model();
//			X[0] += neu[0].v;
//			X[1] += neu[0].m;
//			X[2] += neu[0].h;
//			X[3] += neu[0].n;
//		}
//		t1 = clock();
//
//		fwrite(&s,sizeof(double),1,ffp);
//		double sum_error[4] = { 0 };
//		for (int j = 0; j < 4; j++)
//			sum_error[j] = abs(X0[j] - X[j] / 16.0);// *(X0[j] - X[j] / 16.0);
//
//		//sum_error = sqrt(sum_error);
//		fwrite(&sum_error, sizeof(double), 4, ffp);
//		printf("tt=%d time=%0.3fs\n", tt, double(t1 - t0) / CLOCKS_PER_SEC);
//	}
//	fclose(ffp);
//	Delete();
//
//
//}