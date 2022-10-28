/* TE based on IF model spike trains*/
//-----------------------------------------------------------------------------
//		Comments
//-----------------------------------------------------------------------------



#define _CRT_SECURE_NO_WARNINGS
#include "common_header.h"
#include "def.h"
#include "Read_parameters.h"
#include "New_delete.h"
#include "Kmean.h" 
#include "Compute_Causality.h"
#include "Run_model.h"

int main(int argc, char **argv)
{
    clock_t t0, t1;
    // Config program options:
    po::options_description generic("Generic Options");
    generic.add_options()
        ("help,h", "produce help message")
        ("verbose,v", po::bool_switch(&verbose), "show output")
        ("config,c", po::value<string>()->default_value("NetCau_parameters.ini"), "config file, relative to prefix")
        ;
    po::options_description config("Configs");
    config.add_options()
        ("filename,f", po::value<string>(), "filename of data file")
        ("NE", po::value<int>()->default_value(2), "number of Exc. neurons")
        ("NI", po::value<int>()->default_value(0), "number of Inh. neurons")
        ("order", po::value<string>()->default_value("1 1"), "order of TE. Syntax: 'order_x order_y'; ")
        ("T_Max", po::value<double>()->default_value(1e7), "Maximum length of time series.")
        ("DT", po::value<double>()->default_value(2e4), "Minimum length of time series.")
        ("auto_T_max", po::value<int>()->default_value(1), "")
        ("bin", po::value<double>()->default_value(0.5), "time bin for causality calculation.")
        ("sample_delay", po::value<double>()->default_value(0.0), "tau := time delay")
        ("shuffle,s", po::bool_switch(&shuffle_flag), "shuffle the raw spike train for permutation test.")
        ("matrix_name", po::value<string>()->default_value(""), "filename of connectivity matrix.")
        ("path_input", po::value<string>()->default_value(""), "path for input data file")
        ("path_output", po::value<string>()->default_value(""), "path for output data file")
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
    // existing directory for output files;
    // string dir;
    // dir = vm["prefix"].as<string>();

    Read_parameters(vm);
    Output_filename();

    Run_model();

    return 0;
}