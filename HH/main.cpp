
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

int main(int argc,char **argv) {
	long seed_conn, seed_dym;
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
    ("config,c", po::value<string>()->default_value("./HH/NetModel_parameters.ini"), "config filename.")
    ;
  po::options_description config("Configs");
  config.add_options()
    ("NE",          po::value<int>()->default_value(2), "num of E neurons")
    ("NI",          po::value<int>()->default_value(0), "num of I neurons")
    ("seed",        po::value<string>()->default_value("11 11"), "seed to generate connectivity matrix and init Poisson generators.")
    ("TrialID",     po::value<int>()->default_value(0), "Default: 0. for multiple trials with fixed CS and change Poisson seeds")
    ("T_Max",       po::value<double>()->default_value(1e7), "Simulation time period, unit ms.")
    ("T_step",      po::value<double>()->default_value(0.2), "Time step, unit ms.")
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
    ("save_mode",   po::value<string>()->default_value("w"), "'a' for append, 'w' for write by overwrite original data file")
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

  vector<int> seed_buff;
	str2vec(vm["seed"].as<string>(), seed_buff);
	seed_conn = seed_buff[0];  // Create connect matrix
	seed_dym  = seed_buff[1];  // Initialization & Poisson
  std::mt19937 rng_conn(seed_conn), rng_dym(seed_dym);

	Read_parameters(vm);
	out_put_filename();
	Initialization(rng_conn, rng_dym);

  int T_Max_current_run = T_Max;
	t0 = clock();
	if (Lyapunov)
		MLE = Largest_Lyapunov(seed_dym, 1, T_step);
	else
		Run_model();

	// save neuronal states
	SaveNeuronState();

	double total_fire_num[2] = { 0 };
	for (int i = 0; i < N; i++)
		total_fire_num[i < NE ? 0 : 1] += neu[i].fire_num;

	mean_fire_rate = (total_fire_num[0] + total_fire_num[1]) / T_Max_current_run * 1000 / N; //(Hz)
	printf("mean rate (Hz) = %0.2f ", mean_fire_rate);
	printf("(E : %.3f, I : %.3f)\n", total_fire_num[0] / T_Max * 1000 / NE, total_fire_num[1] / T_Max * 1000 / NI);

	t1 = clock();
	printf("Total time = %0.3fs \n\n", double(t1 - t0) / CLOCKS_PER_SEC);
	Delete();
}