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
		a[i].v = b[i].v + u[id] / s*e0;
		a[i].n = b[i].n + u[id + 1] / s*e0;
	}
}

double Largest_Lyapunov(long &seed, double dt,double h)
{
	struct neuron *neu_per, *neu_per_old;
	neu_per = new struct neuron[N];
	neu_per_old = new struct neuron[N];
	double *multiply_index = new double[N];

	int k = 2;   // v,n
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


	for (int i = 0; i < N; i++)
	{
		Exchange(neu_per[i], neu[i]);
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

				for (int i = 0; i < N; i++)
					Generate_Poisson_times(neu[i], t, h);

				evolve_model_with_initial_timestep(neu, neu_old, t, h);
				evolve_model_with_initial_timestep(neu_per, neu_per_old, t, h);

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
	
			for (int i = 0; i < N; i++)
			{
				int id = i * k;

				u[id] = neu_per[i].v - neu[i].v;
				u[id + 1] = neu_per[i].n - neu[i].n;
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
