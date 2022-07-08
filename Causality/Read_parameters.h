void Read_parameters(po::variables_map& vm)
{
	char ch[100];

	NE = vm["NE"].as<int>();
	NI = vm["NI"].as<int>();
	N = NE + NI;
	T_Max = vm["T_Max"].as<double>();
	DT = vm["DT"].as<double>();
	if (DT <= 1e5)
		DT = 1e5;
	auto_T_max = vm["auto_T_max"].as<int>();


	bin = vm["bin"].as<double>();
    vector<int> order_vec;
	str2vec(vm["order"].as<string>(), order_vec);
	for (int i = 0; i < 2; i++)
		order[i] = order_vec[i];

	delay = vm["sample_delay"].as<double>();

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

	sprintf(ch, "%d", N), strcat(path_input, ch), strcat(path_input, "/");
	sprintf(ch, "%d", N), strcat(path_output, ch), strcat(path_output, "/");
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
	strcat(str, "T="), sprintf(ch, "%0.0e", T_Max), strcat(str, ch);
	strcat(str, "-"), strcat(str, input_filename);

	//strcat(str, "T"), sprintf(ch, "%d", int(T_Max / DT + 0.5)), strcat(str, ch);
	strcat(str, ".dat");

	strcpy(output_filename, str);
}