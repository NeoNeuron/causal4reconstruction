double multiply_vector(double *a, double *b, int n)
{
	double s = 0;
	for (int i = 0; i < n; i++)
		s += a[i] * b[i];
	return s;
}

double norm2(double *a, int n)
{
	return sqrt(multiply_vector(a, a, n));
}

// a<--b, k stands for the number in each neuron having perturbation 

void Initial_perturbe_trace(double t, struct neuron *a, struct neuron *b, double *u, int k, double e0, double *multiply_index)
{
	int n = N*k;  // the length in u
	double s = norm2(u, n);

	for (int i = 0; i < N; i++)
	{
		int id = i * k;

		if (Lib_method  && (t - b[i].last_fire_time < T_ref) && (t - a[i].last_fire_time < T_ref))
		{
			multiply_index[i] *= e0 / s;
		}
		else
		{
			a[i].v = b[i].v + u[id] / s*e0;
			a[i].m = b[i].m + u[id + 1] / s*e0;
			a[i].h = b[i].h + u[id + 2] / s*e0;
			a[i].n = b[i].n + u[id + 3] / s*e0;
		}

		a[i].G_se = b[i].G_se + u[id + 4] / s*e0;
		a[i].G_sse = b[i].G_sse + u[id + 5] / s*e0;

		a[i].G_si = b[i].G_si;
	}
}

double Largest_Lyapunov(long &seed, double dt,double h)
{
	struct neuron *neu_per, *neu_per_old;
	neu_per = new struct neuron[N];
	neu_per_old = new struct neuron[N];
	double *multiply_index = new double[N];

	int k = 6;   // v,m,h,n,G_se,G_sse
	int n = N * k;   
	double *u = new double[n];

	double t = 0, tt = 0, e0 = 1.0e-6;   //edit
	double t_test = 0;
	int step = 0, max_step, m;
	
	double lyapunov = 0, lyapunov_old;
	double sum_log = 0;
	int lyapunov_increment = int(1000/dt+0.1);

	m = int(dt / h + 0.1);
	max_step = int(T_Max / dt + 0.1);/////////////////

	//t = -1000;
	//while (t + T_Step_Large <= 0)
	//{
	//	if (Nu > Epsilon)
	//		for (int i = 0; i < N; i++)
	//			Generate_Poisson_times(neu[i], t, T_Step_Large);
	//	if (Lib_method == 1)
	//		run_lib_method(neu, neu_old, t, T_Step_Large);
	//	else
	//		run_regular_method(neu, neu_old, t, T_Step_Large);
	//	t += T_Step_Large;
	//}

	for (int i = 0; i < N; i++)
	{
		Exchange(neu_per[i], neu[i]);
		neu_per[i].state = 0;
		neu_per[i].fire_num = 0;
		neu_per[i].wait_strength_E = 0;
		neu_per[i].wait_strength_I = 0;
	}

	for (int i = 0; i < N; i++)
		multiply_index[i] = 1;

	for (int i = 0; i < n; i++)
		u[i] =  (Random(seed) - 0.5);
	Initial_perturbe_trace(t, neu_per, neu, u, k, e0, multiply_index);
	


	while (step < max_step)
	{
		lyapunov_old = lyapunov;

		for (int ii = 0; ii < lyapunov_increment; ii++)
		{
			for (int i = 0; i < m; i++)
			{	
				if (Nu > Epsilon)
					for (int j = 0; j < N; j++)
						Generate_Poisson_times(neu[j], t, h);

				for (int j = 0; j < N; j++)
				{
					evolve_model_with_correct_timestep(j, neu[j], t, h);
					evolve_model_with_correct_timestep(j, neu_per[j], t, h);
				}
				Update_Conductance(t + h, neu);
				Update_Conductance(t + h, neu_per);
				t += h;
			}
			step++;



			//// to make sure that both reference and perturbed neuron are in or out of refractory period
			int each_pare_in_fine_state = 1;     
			for (int i = 0; i < N; i++)
			{
				if ((t - neu[i].last_fire_time < T_ref) && (t - neu_per[i].last_fire_time >= T_ref))
				{
					each_pare_in_fine_state = 0;
					break;
				}
				else if ((t - neu[i].last_fire_time >= T_ref) && (t - neu_per[i].last_fire_time < T_ref))
				{
					each_pare_in_fine_state = 0;
					break;
				}

			}
			if (!each_pare_in_fine_state)
				continue;
	
			//// when neuron number is small, we just wait until no neuron in refractory period
			if (Lib_method  && N <= 5)
			{
				int all_neu_out_refractory = 1;
				for (int i = 0; i < N; i++)
				{
					if (t - neu[i].last_fire_time < T_ref || t - neu_per[i].last_fire_time < T_ref)
					{
						all_neu_out_refractory = 0;
						break;
					}
				}

				if (!all_neu_out_refractory)
					continue;
			}


			for (int i = 0; i < N; i++)
			{
				int id = i * k;
				if (Lib_method  && t - neu[i].last_fire_time < T_ref && t - neu_per[i].last_fire_time < T_ref)
				{
					u[id] = 0;
					u[id + 1] = 0;
					u[id + 2] = 0;
					u[id + 3] = 0;
				}
				else
				{
					u[id] = neu_per[i].v - neu[i].v;
					u[id + 1] = neu_per[i].m - neu[i].m;
					u[id + 2] = neu_per[i].h - neu[i].h;
					u[id + 3] = neu_per[i].n - neu[i].n;
				}

				u[id + 4] = neu_per[i].G_se - neu[i].G_se;
				u[id + 5] = neu_per[i].G_sse - neu[i].G_sse;

			}

			for (int i = 0; i < N; i++)           /// modify voltage
			{			
				if (Lib_method )
				{					
					int t_id, t_id_per;
					t_id = int((neu[i].last_fire_time + T_ref) / dt + 1);
					t_id_per = int((neu_per[i].last_fire_time + T_ref) / dt + 1);
				
					if (step == MAX(t_id, t_id_per))         /////////////
					{
						int id = i*k;
						u[id] *= multiply_index[i];
						u[id + 1] *= multiply_index[i];
						u[id + 2] *= multiply_index[i];
						u[id + 3] *= multiply_index[i];

						multiply_index[i] = 1;
					}
				}
			}

			sum_log += log(norm2(u, n) / e0);
			Initial_perturbe_trace(t, neu_per, neu, u, k, e0, multiply_index);

		}

		lyapunov = sum_log / t;	
	
		if (abs(lyapunov - lyapunov_old) < 1e-4 && t >= T_Max)
			break;


		if (t - tt >= 1000)
		{
			tt = t;
			printf("t=%0.2f MLE=%f\n", t, lyapunov);			
		}
	}

	printf("t=%0.2e MLE is %f\n", t, lyapunov);

	delete[]neu_per, delete[]neu_per_old, delete[]u, delete[]multiply_index;
	return lyapunov;
}
