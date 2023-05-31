/* TE based on IF model spike trains*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS
#include "common_header.h"
#include "Kmean.h" 
#include "Compute_Causality.h"
#include "Run_model.h"
std::random_device rd;
std::mt19937 rng(rd());

int main(int argc, char **argv)
{
    bool verbose = false;
    bool shuffle_flag = false;
    // Config program options:
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
        ("verbose,v", po::bool_switch(&verbose), "show output")
        ("config,c", po::value<string>()->default_value("./Causality/NetCau_parameters.ini"), "config file, relative to prefix")
        ;
    po::options_description data_structure(
		R"(Data structure of output file:
	The original data file contains N*N*(k+12)+9 double-type values.
	The first N*N*(k+12) values are related to pairwise inferenced causality values, which are organized as follows:
	For each pair of neurons (y,x), the following values are stored:
		TE value, idx of y, idx of x, p(x=1), p(y=1), p(x = 1 | x- = 0, y- = 0),
		\delta p_{y->x}: dp1,dp2,...,dpk,
		\Delta p_m = p(x = 1,y- = 1)/(p(x = 1)p(y- = 1)) - 1,
		te_order5, GC value, sum of TDMI over delays, sum of TDCC^2 over delays,
		appxo for 2sumDMI.
	The last 9 values are the length of time sereis (L), threshold for reconstruction based on 4 causality measures, reconstruction accuracy based on 4 causality measures.)"
	);
    po::options_description config("Configs");
    config.add_options()
        ("filename,f", po::value<string>(), "filename of data file")
        ("NE", po::value<int>()->default_value(2), "number of Exc. neurons")
        ("NI", po::value<int>()->default_value(0), "number of Inh. neurons")
        ("order", po::value<string>()->default_value("1 1"), "order of TE. Syntax: 'order_x order_y'; ")
        ("T_Max", po::value<double>()->default_value(1e7), "Maximum length of time series.")
        ("DT", po::value<double>()->default_value(1e5), "Minimum length of time series.")
        ("auto_T_max", po::value<int>()->default_value(1), "auto-adjust T_Max to the maximum time of the spike events.")
        ("bin", po::value<double>()->default_value(0.5), "time bin for causality calculation.")
        ("sample_delay", po::value<double>()->default_value(0.0), "tau := time delay")
        ("shuffle,s", po::bool_switch(&shuffle_flag), "shuffle the raw spike train for permutation test.")
        ("matrix_name", po::value<string>()->default_value("./"), "filename of connectivity matrix.")
        ("path_input", po::value<string>()->default_value("./"), "path for input data file")
        ("path_output", po::value<string>()->default_value("./"), "path for output data file")
        ("n_thread,j", po::value<int>()->default_value(1), "number of threads for causality estimation")
        ;
    // create variable map
    po::variables_map vm;
    po::options_description cml_options;
    cml_options.add(generic).add(config);
    po::store(po::parse_command_line(argc, argv, cml_options), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << generic << '\n';
        cout << data_structure << '\n';
        cout << config << '\n';
        return 1;
    }
    // loading parsers from config file
    ifstream config_file;
    if (vm.count("config")) {
        string cfname = vm["config"].as<string>();
        config_file.open(cfname.c_str());
        if (!config_file) {
            cout << "[WARNNING]: config file '" << cfname << "' does not exist! Run with default configs and CML arguments" << endl;
        } else {
            po::store(po::parse_config_file(config_file, config), vm);
            po::notify(vm);
        }
    }
    // Override config params with cml params
    po::store(po::parse_command_line(argc, argv, cml_options), vm);
    po::notify(vm);
    // existing directory for output files;
    // string dir;
    // dir = vm["prefix"].as<string>();

    // define variables
    int num_threads_openmp = vm["n_thread"].as<int>();


    // --------------------------------------
	int NE = vm["NE"].as<int>();	// number of Exc. neurons
	int NI = vm["NI"].as<int>();	// number of Inh. neurons
    int N = NE + NI;				// total neuron number
	double T_Max = vm["T_Max"].as<double>();	// total T_Max
	double DT    = vm["DT"].as<double>();		// division into DT length
	int auto_T_max = vm["auto_T_max"].as<int>();
	double bin = vm["bin"].as<double>();		// bin size
    int order[3];			// order x , order y , max order +1
    int k;					// k = order_y
	double delay = vm["sample_delay"].as<double>();	
		// delay: x(t+delay+bin), x(t+delay), y(t)
    int tau = int(delay / bin);
		// tau = delay/bin,  delay: x(n+tau+1), x(n+tau), y(n)
    double L, accuracy[4], threshold[4];

    char input_filename[200], output_filename[200], matrix_name[100];
    char path_input[200], path_output[200];
    int m[2];		// pow(2.0,order_x) and pow(2,order_y)

    FILE *FP;

    vector<int> order_vec;
	str2vec(vm["order"].as<string>(), order_vec);
	for (int i = 0; i < 2; i++)
		order[i] = order_vec[i];

	strcpy(input_filename,  vm["filename"].as<string>().c_str());
	strcpy(matrix_name,  vm["matrix_name"].as<string>().c_str());
	strcpy(path_input,    vm["path_input"].as<string>().c_str());
	strcpy(path_output,  vm["path_output"].as<string>().c_str());

	if (N == NE) {
		strcat(path_input,  "EE/N=");
		strcat(path_output, "EE/N=");
	} else if (N == NI) {
		strcat(path_input,  "II/N=");
		strcat(path_output, "II/N=");
	} else {
		strcat(path_input,  "EI/N=");
		strcat(path_output, "EI/N=");
	}

    char ch[100];
	sprintf(ch, "%d", N), strcat(path_input, ch), strcat(path_input, "/");
	sprintf(ch, "%d", N), strcat(path_output, ch), strcat(path_output, "/");

    // Output_filename();
	k = order[1];
	m[0] = int(pow(2.0, order[0]));
	m[1] = int(pow(2.0, order[1]));
	order[2] = order[0] > order[1] ? order[0] : order[1];
	order[2]++;

	char str[200];

	strcpy(str, path_output);

	strcat(str, "TGIC2.0-K="), sprintf(ch, "%d", order[0]), strcat(str, ch);
	strcat(str, "_"), sprintf(ch, "%d", order[1]), strcat(str, ch);

	strcat(str, "bin="), sprintf(ch, "%0.2f", bin), strcat(str, ch);
	strcat(str, "delay="), sprintf(ch, "%0.2f", delay), strcat(str, ch);
	strcat(str, "T="), sprintf(ch, "%.2e", T_Max), strcat(str, ch);
	strcat(str, "-"), strcat(str, input_filename);

	//strcat(str, "T"), sprintf(ch, "%d", int(T_Max / DT + 0.5)), strcat(str, ch);
	if (shuffle_flag)
		strcat(str, "_shuffle");
	strcat(str, ".dat");

	strcpy(output_filename, str);

    // Run_model();
    int read_repeat, data_length, interval;
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
		T_Max = Find_T_Max(fp, verbose); // find the data time T_Max

	if (T_Max / DT <= 0.99)
	{
		printf("Warning! T_max=%0.3e < DT=%0.3e!\n", T_Max, DT);
		printf("path:%s\nfilename:%s\n", path_input, input_filename);
		T_Max = DT;
	}
	read_repeat = int(T_Max / DT + 0.01);

	data_length = int(DT / bin);

    // X[id][0/1]  | size=(N, DT/bin+1) | partial binarized spike train
	vector<vector<unsigned int> > X(N, vector<unsigned int>(data_length, 0));

    // z[i-->j][p] | size=(N*N, 2**(1+order[0]+order[1]) | pair wise TE's p(x,x-,y-)
	vector<vector<double> > z(N*N, vector<double>(2 * m[0] * m[1], 0));

	for (int id = 0; id < read_repeat; id++)
	{
		read_data(fp, 1.0*id*data_length*bin, 1.0*(1 + id)*data_length*bin, bin, X, N);
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
			compute_p(X, y, x, N, z, tau, order, m);
		}
	}
	fclose(fp);

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

    // TE[N][N], GC, sum of DMI, sum of NCC^2 y-->x	
    // 2TE,2DMI, [N][N], y-->x
	vector<vector<double> > TE(N, vector<double>(N, 0));
	vector<vector<double> > GC(N, vector<double>(N, 0));
	vector<vector<double> > DMI(N, vector<double>(N, 0));
	vector<vector<double> > NCC(N, vector<double>(N, 0));
	vector<vector<double> > TE_2(N, vector<double>(N, 0));
	vector<vector<double> > DMI_2(N, vector<double>(N, 0));
	compute_causality(z, order, m, N, FP, TE, GC, DMI, NCC, TE_2, DMI_2);

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

    // conn_file_fname
	strcpy(str, path_input), strcat(str, matrix_name);

	// strcpy(str, path_input);
	// int id_p1 = 0, id_p2 = 0;
	// for (int i = 0; i < strlen(input_filename); i++)
	// {
	// 	if (input_filename[i] == 'p')
	// 		id_p1 = i;
	// 	if (input_filename[i] == 's' && i - id_p1 < 8)
	// 		id_p2 = i;
	// }
	// strcpy(matrix_name, "connect_matrix-");
	// char chh[10] = "";
	// for (int i = id_p1; i < id_p2; i++)
	// 	chh[i - id_p1] = input_filename[i];
	// strcat(matrix_name, chh), strcat(matrix_name, "0.dat");
	// strcat(str, matrix_name);
	// printf("conn_filename=%s\n", str);

	//  reconstruction & kmean	
	Kmean(TE_2, threshold[0], accuracy[0], str, L, NE, NI);
	Kmean(GC, threshold[1], accuracy[1], str, L, NE, NI);
	Kmean(DMI_2, threshold[2], accuracy[2], str, L, NE, NI);
	Kmean(NCC, threshold[3], accuracy[3], str, L, NE, NI);
	fwrite(&L, sizeof(double), 1, FP);
	fwrite(&threshold, sizeof(double), 4, FP);
	fwrite(&accuracy, sizeof(double), 4, FP);

	fclose(FP);

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

    return 0;
}