/* TE based on IF model spike trains*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS
#include <chrono>
#include "common_header.h"
int num_threads_openmp = 1;
#include "Compute_Causality.h"
#include "Run_model.h"
std::random_device rd;
std::mt19937 rng(rd());

int main(int argc, char **argv)
{
    bool verbose = false;
    bool shuffle_flag = false;
	bool auto_Tmax = false;
    // Config program options:
    po::options_description generic("Generic options");
    generic.add_options()
        ("help,h", "produce help message")
        ("verbose,v", po::bool_switch(&verbose), "show output")
        ("config,c", po::value<string>(), "config file (*.ini), relative to prefix")
        ;
    po::options_description data_structure(
		R"(Data structure of output file:
	The original data file contains N*N*(l+12) double-type values of 
	pairwise inferenced causality values, which are organized as follows:
	For each pair of neurons (y,x), the following values are stored:
		TE value, idx of y, idx of x, p(x=1), p(y=1), p(x = 1 | x- = 0, y- = 0),
		\delta p_{y->x}: dp1,dp2,...,dpl,
		\Delta p_m = p(x = 1,y- = 1)/(p(x = 1)p(y- = 1)) - 1,
		te_order5, GC value, sum of TDMI over delays, sum of TDCC^2 over delays,
		appxo for 2sumDMI.)"
	);
    po::options_description config("Configs");
    config.add_options()
        ("spike_train_filename,f", po::value<string>(), "str, filename of spike train data file")
        ("N,N", po::value<int>()->default_value(2), "int, number of neurons")
        ("order", po::value<string>()->default_value("1 1"), "tuple, order of TE. Syntax: 'order_x order_y', 'k l'; ")
        ("Tmax,T", po::value<double>()->default_value(1e7), "float, maximum time length used in estimation, default unit in ms.")
        ("DT", po::value<double>()->default_value(1e5), "float, minimum length of time series, same unit as Tmax.")
        ("auto_Tmax", po::bool_switch(&auto_Tmax), "bool, auto-adjust Tmax to the maximum time of the spike train data.")
        ("dt", po::value<double>()->default_value(0.5), "float, temporal binsize to binarize spike trains, same unit as Tmax.")
        ("delay", po::value<double>()->default_value(0.0), "float, time delay, tau, same unit as Tmax.")
        ("shuffle,s", po::bool_switch(&shuffle_flag), "bool, shuffle the raw spike train for permutation test.")
        ("path,p", po::value<string>()->default_value("./"), "str, path for spike train, masking and output files")
        ("n_thread,j", po::value<int>()->default_value(1), "int, number of threads for causality estimation")
        ("mask_file", po::value<string>(), "str, filename of mask file in compressed sparse matrix.")
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
    num_threads_openmp = vm["n_thread"].as<int>();


    // --------------------------------------
    int N = vm["N"].as<int>();				// total neuron number
	double Tmax = vm["Tmax"].as<double>();	// total Tmax
	double DT = vm["DT"].as<double>();		// division into DT length
	double dt = vm["dt"].as<double>();		// temporal binsize
    int order[3];			// order x , order y , max order +1
    int k;					// k = order_y
	double delay = vm["delay"].as<double>();	
		// delay: x(t+delay+dt), x(t+delay), y(t)
    int tau = int(delay / dt);
		// tau = delay/dt,  delay: x(n+tau+1), x(n+tau), y(n)
    double L, accuracy[4], threshold[4];

    char input_filename[200];
    char path[200];
    int m[2];		// pow(2.0,order_x) and pow(2,order_y)

    FILE *FP;

    vector<int> order_vec;
	str2vec(vm["order"].as<string>(), order_vec);
	for (int i = 0; i < 2; i++)
		order[i] = order_vec[i];

	strcpy(input_filename, vm["spike_train_filename"].as<string>().c_str());
	strcpy(path, vm["path"].as<string>().c_str());

	// naming output folder 
	strcat(path, "/");

    // Output_filename();
	k = order[1];
	m[0] = int(pow(2.0, order[0]));
	m[1] = int(pow(2.0, order[1]));
	order[2] = order[0] > order[1] ? order[0] : order[1];
	order[2]++;

	char str[500], buffer[100];

    // Run_model();
    int read_repeat, data_length, interval;
	FILE *fp;

	auto ct0 = chrono::steady_clock::now();
	clock_t t0 = clock();

	strcpy(str, path), strcat(str, input_filename), strcat(str, "_spike_train.dat");
	fp = fopen(str, "rb");
	if (fp == NULL)
	{
		printf("Error in read file: %s\n%s\n", path, input_filename);
		exit(0);
	}

	//	 compute the probability distribution	
	if(auto_Tmax)
		Tmax = Find_Tmax(fp, verbose); // find the data time Tmax

	if (Tmax / DT <= 0.99)
	{
		printf("Warning! Tmax=%0.3e < DT=%0.3e!\n", Tmax, DT);
		printf("path:%s\nfilename:%s\n", path, input_filename);
		Tmax = DT;
	}
	read_repeat = int(Tmax / DT + 0.01);

	// most memory cost
	data_length = int(DT / dt);

	char mask_fname[200];
	bool mask_toggle = false;
	vector<vector<double> > mask_indices(2);
    if (vm.count("mask_file")) {
		strcpy(mask_fname, path); 
		strcat(mask_fname, vm["mask_file"].as<string>().c_str());
		FILE *fp_mask;
		fp_mask = fopen(mask_fname, "rb");
		if (fp_mask == NULL)
		{
			printf("Warning! can not open connect matrix file! %s \n", mask_fname);
			//	exit(0);
		}

		long int file_size;
		double s;
		fseek(fp_mask, 0, SEEK_END);
		file_size = ftell(fp_mask)/sizeof(double);
		fseek(fp_mask, 0, SEEK_SET);

		for (int i = 0; i < 2; i++) {
			mask_indices[i].resize(file_size/2, 0);
			fread(&mask_indices[i][0], sizeof(double), file_size/2, fp_mask);	
		}
		std::fclose(fp_mask);
		mask_toggle = true;
		if (verbose)
			printf("mask on, and mask_size=%d\n", mask_indices[0].size());
	}

    // X[id][0/1]  | size=(N, DT/dt+1) | partial binarized spike train
	vector<vector<unsigned short int> > X(N, vector<unsigned short int>(data_length, 0));

    // z[i-->j][p] | size=(N*N, 2**(1+order[0]+order[1]) | pair wise TE's p(x,x-,y-)
	vector<vector<double> > z;
	if (mask_toggle)
		z.resize(mask_indices[0].size(), vector<double>(2 * m[0] * m[1], 0));
	else
		z.resize(N*N, vector<double>(2 * m[0] * m[1], 0));

	int num_pairs = mask_toggle ? mask_indices[0].size() : N * N;
	for (int id = 0; id < read_repeat; id++)
	{
		read_data(fp, 1.0*id*data_length*dt, 1.0*(1 + id)*data_length*dt, dt, X, N);
		//Notice!  id*data_length may larger than MAX INT 2147483647

		// shuffling data
		if (shuffle_flag) {
			for (int neu_id=0; neu_id < N; neu_id ++)
				shuffle(&X[neu_id][0], &X[neu_id][data_length], rng);
		}

		#pragma omp parallel for num_threads(num_threads_openmp)
		for (int i = 0; i < num_pairs; i++) {  // y-->x
			int x, y;
			if (mask_toggle)	// y: pre, x: post
				y = mask_indices[0][i], x = mask_indices[1][i];
			else
				y = i / N, x = i % N;
			// Comment below the calculate auto-correlation
			if (y == x)
				continue;
			compute_p(X, y, x, N, z, tau, order, m, i);
		}
		#pragma omp parallel for num_threads(num_threads_openmp)
		for (int i = 0; i < N; i++)
			X[i].assign(data_length, 0);
	}
	std::fclose(fp);

	// Count the number of total events
	L = 0;
	if (mask_toggle) {
		for (int i = 0; i < 2 * m[0] * m[1]; i++)
			L += z[0][i];
	} else {
		for (int i = 0; i < 2 * m[0] * m[1]; i++)
			L += z[1][i];
	}

	for (auto i = z.begin(); i != z.end(); i++)
		for (auto j = (*i).begin(); j != (*i).end(); j++)
			*j /= L;

	auto ct1 = chrono::steady_clock::now();
	clock_t t1 = clock();
	// compute TE(x_n+1+tau,x_n+tau,y_n), GC(x_n+1+tau,x_n+tau,y_n) 
	// sum DMI(x_n+1+tau,y_n), CC(x_n+1+tau,y_n)

	// format output filename
    char output_filename[200];
	sprintf(buffer, "TGIC2.0-K=%d_%dbin=%0.2fdelay=%0.2fT=%0.2e-", order[0], order[1], dt, delay, Tmax);
	strcpy(str, path);
	strcat(str, buffer);
	strcat(str, input_filename);
	if (shuffle_flag)
		strcat(str, "_shuffle");
	strcat(str, ".dat");
	strcpy(output_filename, str);

	FP = fopen(output_filename, "wb");
	if (verbose)
		printf("save to file :%s\n\n", output_filename);

    // TE, GC, sum of DMI, sum of NCC^2 y-->x	
	compute_causality(z, order, m, N, FP, mask_toggle, mask_indices);

	std::fclose(FP);

	auto ct2 = chrono::steady_clock::now();
	clock_t t2 = clock();
	if (verbose) {
		std::chrono::duration<double> elapsed_seconds;
		printf("Tmax=%0.2e L=%0.2e\n\n", Tmax, L);
        elapsed_seconds = ct1 - ct0;
		printf(">> preprocessing time: %0.3f s (CPU time: %0.3f s)\n", elapsed_seconds, double(t1 - t0) / CLOCKS_PER_SEC);
        elapsed_seconds = ct2 - ct1;
		printf(">> estimation time:    %0.3f s (CPU time: %0.3f s)\n", elapsed_seconds, double(t2 - t1) / CLOCKS_PER_SEC);
        elapsed_seconds = ct2 - ct0;
		printf(">> total time:         %0.3f s (CPU time: %0.3f s)\n", elapsed_seconds, double(t2 - t0) / CLOCKS_PER_SEC);
    }

    return 0;
}