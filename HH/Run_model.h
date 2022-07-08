// evolve for one single time step from t to t+dt
void evolve_model_with_correct_timestep(int n, struct neuron &a, double t, double dt)
{
	void Update_Lib_way(struct neuron &neu_a, int interpolation_order, int input_dim, int output_dim);

	if (dt < 0)
	{
		printf("Warning! dt in evolve_model_with_correct_timestep < 0 !! dt = %e\n", dt);
		printf("n=%d t=%f state=%d\n", n, t, a.state);
		exit(0);
	}

	if (Regular_method)
		Update_RK2(n, a, t, dt);
	else if (Lib_method)
	{
		double local_t = a.last_fire_time + T_ref;
		if (local_t <= t)
		{
			Update_RK2(n, a, t, dt);

			if (Lib_method && a.if_fired && (Power_spectrum || record_data[1]))
			{
				a.last_hit[0] = a.I_input;
				a.last_hit[1] = a.m;
				a.last_hit[2] = a.h;
				a.last_hit[3] = a.n;
			}

			if (Lib_method && a.if_fired)
				Update_Lib_way(a, IntOrd, 4, 4);

		}
		else if (local_t > t + dt)
			Update_neu_G(n, a, t, dt);
		else  // t < local_t < t+dt case
		{
			Update_neu_G(n, a, t, local_t - t);
			Update_RK2(n, a, local_t, t + dt - local_t);
		}
	}
	else if (ETD_method)
		Update_ETD2(n, a, t, dt);
	else
		Update_ETDRK2(n, a, t, dt);
}


void Exchange(struct neuron &a, struct neuron b)  // a <-- b
{
	a = b;
	a.if_fired = 0;
	a.state = 0;
	if (ETD_method)
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 9; j++)
				a.F[i][j] = b.F[i][j];
		a.id_F = b.id_F;
	}
}

//-----------------------------------------------------------------------------
//		Evolve single neuron from last hit time t to t+T_ref with library method (v,m,h,n)
//      Multi interpolation with interpolation_order (1--linear,2--quadratic)
//		Multi-interpolation, input dimension, outout dimension
//-----------------------------------------------------------------------------

void Update_Lib_way(struct neuron &neu_a, int interpolation_order, int input_dim, int output_dim)
{
	// I_input,m,h,n ____ 4-digit number like decimalism but with each lenth l[4] not 10    
	// l--length in each I,m,h,n. this is actually Lib_resolution
	// a--each minimin id for interpolation in 4-D. 
	// b--each id for interpolation in 4-D. scan id
	// c--each change in id id for interpolation in 4-D. scan change

	int *l;  
	double *input, *output; //I,m,h,n & v,m,h,n
	int *a,*b, *c;

	input = new double[input_dim]; output = new double[output_dim];
	l = new int[input_dim]; a = new int[input_dim];
	b = new int[input_dim]; c = new int[input_dim];

	for (int i = 0; i < input_dim; i++)
		output[i] = 0;

	for (int i = 0; i < input_dim; i++)
		l[i] = Lib_resolution[i];

	input[0] = neu_a.I_input;
	input[1] = neu_a.m;
	input[2] = neu_a.h;
	input[3] = neu_a.n;

	for (int i = 0; i < input_dim; i++)
	{
		a[i] = int((input[i] - Lib_unique[i][0]) / (Lib_unique[i][1] - Lib_unique[i][0]) + 0.1);
		if (a[i] < 0)
		{
			printf("Warning! a%d=%d\n", i, a[i]);
			a[i] = 0;
		}
		else if (a[i] > l[i] - interpolation_order - 1)
		{
			printf("Warning! a%d=%d max_length=%d t=%0.3f\n ", i, a[i], Lib_resolution[i], neu_a.t);
			a[i] = l[i] - interpolation_order - 1;
		}
	}


	for (int i = 0; i < int(pow((interpolation_order + 1), input_dim) + 0.05); i++)
	{
		int k;
		double s = 1;
		k = i;
		for (int j = 0; j < input_dim; j++)
		{
			c[j] = k % (interpolation_order + 1);
			k /= (interpolation_order + 1);
			for (int ll = 0; ll < (interpolation_order + 1); ll++)
			{
				if (ll == c[j])
					continue;
				else
					s *= (input[input_dim - 1 - j] - Lib_unique[input_dim - 1 - j][a[input_dim - 1 - j] + ll]) /
					(Lib_unique[input_dim - 1 - j][a[input_dim - 1 - j] + c[j]] - Lib_unique[input_dim - 1 - j][a[input_dim - 1 - j] + ll]);
			}
		}
		for (int j = 0; j < input_dim; j++)
			b[j] = a[j] + c[input_dim - 1 - j];

		int m = b[0];
		for (int j = 1; j < input_dim; j++)
			m = m*l[j] + b[j];

		for (int j = 0; j < output_dim; j++)
			output[j] += Lib_data[j + input_dim][m] * s;
	}

	neu_a.v = output[0];
	neu_a.m = output[1];
	neu_a.h = output[2];
	neu_a.n = output[3];
//	neu_a.t += T_ref;
	delete[]l, delete[]a, delete[]input, delete[]output, delete[]b, delete[]c;

}

//-----------------------------------------------------------------------------
//		To record v trace in Power spectrum in Lib_method
//-----------------------------------------------------------------------------

void Find_v_in_lib_method(struct neuron neu_a, int interpolation_order, int input_dim, double &output, double dt)
{
	// I_input,m,h,n ____ 4-digit number like decimalism but with each lenth l[4] not 10    
	// l--length in each I,m,h,n. this is actually Lib_resolution
	// a--each minimin id for interpolation in 4-D. 
	// b--each id for interpolation in 4-D. scan id
	// c--each change in id id for interpolation in 4-D. scan change

	int t_id = int(dt * 256);  // time_step = 2^-8, 0 <= dt <= T_ref
	double lambda_t = dt * 256 - t_id;

	int *l;
	double *input; //I,m,h,n
	int *a, *b, *c;

	output = 0;

	input = new double[input_dim];
	l = new int[input_dim]; a = new int[input_dim];
	b = new int[input_dim]; c = new int[input_dim];


	for (int i = 0; i < input_dim; i++)
	{
		l[i] = Lib_resolution[i];
		input[i] = neu_a.last_hit[i];
	}

	for (int i = 0; i < input_dim; i++)
	{
		a[i] = int((input[i] - Lib_unique[i][0]) / (Lib_unique[i][1] - Lib_unique[i][0]) + 0.1);
		if (a[i] < 0)
		{
			printf("Warning! a%d=%d\n", i, a[i]);
			a[i] = 0;
		}
		else if (a[i] > l[i] - interpolation_order - 1)
		{
			printf("Warning! a%d=%d max_length=%d t=%0.3f\n", i, a[i], Lib_resolution[i], neu_a.t);
			a[i] = l[i] - interpolation_order - 1;
		}
	}


	for (int i = 0; i < int(pow((interpolation_order + 1), input_dim) + 0.05); i++)
	{
		int k;
		double s = 1;
		k = i;
		for (int j = 0; j < input_dim; j++)
		{
			c[j] = k % (interpolation_order + 1);
			k /= (interpolation_order + 1);
			for (int ll = 0; ll < (interpolation_order + 1); ll++)
			{
				if (ll == c[j])
					continue;
				else
					s *= (input[input_dim - 1 - j] - Lib_unique[input_dim - 1 - j][a[input_dim - 1 - j] + ll]) /
					(Lib_unique[input_dim - 1 - j][a[input_dim - 1 - j] + c[j]] - Lib_unique[input_dim - 1 - j][a[input_dim - 1 - j] + ll]);
			}
		}

		for (int j = 0; j < input_dim; j++)
			b[j] = a[j] + c[input_dim - 1 - j];

		int m = b[0];
		for (int j = 1; j < input_dim; j++)
			m = m*l[j] + b[j];

		double find_v;
		find_v = Lib_v[t_id][m] * (1 - lambda_t) + Lib_v[t_id + 1][m] * lambda_t;
		output += find_v * s;

	}
	delete[]a, delete[]l, delete[]input, delete[]b, delete[]c;

}

void Generate_Poisson_times(struct neuron &a, double t, double dt)
{
	int k = a.Poisson_input_num;
	double t1;

	if (k == -1)
		t1 = a.t - log(1 - Random(a.seed)) / a.Nu; 
	else
		t1 = a.Poisson_input_time[k];   

	k = 0;
	a.Poisson_input_time[k] = t1;
	while (t1 <= t + dt)
	{
		k++;
		t1 += -log(1 - Random(a.seed)) / a.Nu;    // the last one is larger than t+dt

		if (k + 1 > int(T_step * 100 * 2) + 5)
			a.Poisson_input_time = (double *)realloc(a.Poisson_input_time, (k + 20) * sizeof(double));
		a.Poisson_input_time[k] = t1;
	}
	a.Poisson_input_num = k;
}

void Record_Power_spectrum(double t)
{
	fwrite(&t,sizeof(double),1,FP_FFTW);
	for (int i = 0; i < N; i++)
	{
		if (Lib_method && t - neu[i].last_fire_time < T_ref)
		{
			double output;
			Find_v_in_lib_method(neu[i], IntOrd,4,output,t-neu[i].last_fire_time);
			fwrite(&output, sizeof(double), 1, FP_FFTW);
		}
		else
			fwrite(&neu[i].v, sizeof(double), 1, FP_FFTW);
	}
}

void Update_Conductance(double t,struct neuron *a)
{
	for (int i = 0; i < N; i++)  // update G_f G_ff
		for (int j = 0; j < neu[i].Poisson_input_num; j++)
		{
			//neu[i].Poisson_input_time[j] = t;
			double dt = t - neu[i].Poisson_input_time[j]; 
			a[i].G_ff += f[i] * (1 - dt / Sigma_d_E);
			a[i].G_f += f[i] * dt;

			if (CP > 1e-6)
			{
				if (i == 0 && Random(SD) <= CP) // common input
				{
					for (int k = 1; k < N; k++)
					{
						a[k].G_ff += f[i] * (1 - dt / Sigma_d_E);
						a[k].G_f += f[i] * dt;
					}
				}
			}
		}

	for (int k = 0; k < N; k++)
	{
		if (a[k].if_fired)
		{
			a[k].if_fired = 0;
			a[k].fire_num++;

			if (record_data[0] && a[k].state)
			{
				double s = k;
				fwrite(&a[k].last_fire_time, sizeof(double), 1, FP);				
				fwrite(&s, sizeof(double), 1, FP);
			}

			//a[k].last_fire_time = t;
			double dt = t - a[k].last_fire_time;	// update G_se G_si	 
			for (int i = 0; i < N; i++)
				if (k < NE)
				{
					a[i].G_sse += CS[k][i] * (1 - dt / Sigma_d_E);
					a[i].G_se += CS[k][i] * dt;
				}
				else
				{
					a[i].G_ssi += CS[k][i] * (1 - dt / Sigma_d_I);
					a[i].G_si += CS[k][i] * dt;
				}
		}
	}
}

void Run_model()
{
	num_threads_openmp = N >= 8 ? 8 : 4;
	clock_t ts, te, ts0;
	if(ode_type !=2 && ode_type !=4)
	{
		printf("\nError ode_type = %d!!!\n", ode_type);
		exit(0);
	}

	double t = neu[0].t, last_record_t = neu[0].t, tt_fftw = 0, t_lib = 0, t_fp = 0;
	T_Max += neu[0].t;
	double s = -100, ss = neu[0].v;
	double I_th;
	long seed_record_Nu = 140719001;
	for (int i = 0; i < 973; i++)
		Random(seed_record_Nu);

	ts = clock(), ts0 = ts;
	while (t < T_Max)
	{
		if (Nu > Epsilon)
			for (int i = 0; i < N; i++)
				Generate_Poisson_times(neu[i], t, T_step);


		for (int i = 0; i < N; i++)
			evolve_model_with_correct_timestep(i, neu[i], t, T_step);
		Update_Conductance(t + T_step, neu);
		t += T_step;

		te = clock();
		if (double(te - ts) / CLOCKS_PER_SEC >= 10)
		{
			ts = te;			
			double total_fire_num = 0;
			for (int i = 0; i < N; i++)
				total_fire_num += neu[i].fire_num;
			printf("mean rate (Hz) = %0.2f  ", total_fire_num / t * 1e3 / N);
			printf("Time cost is %0.2f s, run time is %0.2f s\n", double(te - ts0) / CLOCKS_PER_SEC, t);
		}

		if (record_data[1] && t > Record_v_start && t <= Record_v_end && t - last_record_t >= 0.5-1e-5) // for library method
		{
			last_record_t = t;
			fwrite(&t, sizeof(double), 1, FP1);
			for (int id = 0; id < N; id++)
				if (!(t - neu[id].last_fire_time <= T_ref && Lib_method))
					fwrite(&neu[id].v, sizeof(double), 1, FP1);
				else if (Lib_method && t - neu[id].last_fire_time <= T_ref)
				{
					double output;
					Find_v_in_lib_method(neu[id], IntOrd, 4, output, t - neu[id].last_fire_time);
					fwrite(&output, sizeof(double), 1, FP1);
				}
		}

		//if (record_data[1] && t > Record_v_start && t <= Record_v_end && t - last_record_t >= 0.5 - 1e-8) 
		//{
		//	last_record_t = t;
		//	fwrite(&t, sizeof(double), 1, FP1);
		//	for (int id = 0; id < N; id++)
		//		if (!(t - neu[id].last_fire_time <= T_ref && Lib_method))
		//		{ 
		//			double s = neu[id].v*neu[id].v;
		//			fwrite(&s, sizeof(double), 1, FP1);
		//		}					
		//		else if (Lib_method && t - neu[id].last_fire_time <= T_ref)
		//		{
		//			double output;
		//			Find_v_in_lib_method(neu[id], IntOrd, 4, output, t - neu[id].last_fire_time);
		//			fwrite(&output, sizeof(double), 1, FP1);
		//		}
		//}

		//if (record_data[1] && t > Record_v_start && t <= Record_v_end && t - last_record_t >= 0.01 - 1e-8) // for library method
		//{
		//	last_record_t = t;
		//	fwrite(&t, sizeof(double), 1, FP1);
		//	fwrite(&neu[0].v, sizeof(double), 1, FP1);
		//	fwrite(&neu[0].m, sizeof(double), 1, FP1);
		//	fwrite(&neu[0].h, sizeof(double), 1, FP1);
		//	fwrite(&neu[0].n, sizeof(double), 1, FP1);
		//	fwrite(&neu[0].dv, sizeof(double), 1, FP1);
		//	fwrite(&neu[0].I_input, sizeof(double), 1, FP1);
		//	double s = neu[0].dv - neu[0].I_input;
		//	fwrite(&s, sizeof(double), 1, FP1);
		//}


		if (Power_spectrum && t > Record_v_start && t <= Record_v_end && t - tt_fftw >= 0.5)
		{
			tt_fftw = t;
			Record_Power_spectrum(t);
		}
		if (RecordFP && t - t_fp >= 10)				// fire pattern 0100101010
		{
			t_fp = t;
			record_fire_pattern(t);
		}

		//Build library, trace of V
		if(library_v_trace && t-t_lib >= 1.0/256)
		{
			t_lib = t;
			fwrite(&neu[0].v, sizeof(double), 1, fp_v);           // record v in library
		}		

		//if (neu[0].if_fired)
		//	I_th = neu[0].I_input;

		//if (t - neu[0].last_fire_time <= T_ref)
		//{
		//	I_CONST = 1;
		//	I_const_input = I_th;
		//}
		//else
		//	I_CONST = 0;

	}


}
