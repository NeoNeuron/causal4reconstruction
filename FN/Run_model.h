/////////////// evolve for one single time step from t to t+dt
void evolve_model_with_correct_timestep(int n, struct neuron &a, double t, double dt)
{	
	if (dt < 0)
	{
		printf("Warning! dt in evolve_model_with_correct_timestep < 0 !! dt = %e\n", dt);
		printf("n=%d t=%f state=%d\n", n, t,a.state);
		exit(0);
	}

	Update_RK4(n, a, t, dt);	
}


/////////////// evolve from t to t+dt 
///////////// there may be times nodes in [t,t+dt], from Poisson and Adaptive time nodes
void Check_update_conti_or_Poisson_input(int n, struct neuron &a, double t, double dt)
{
	if (Nu < Epsilon)
		evolve_model_with_correct_timestep(n, a, t, dt);
	else
	{
		double t1 = t;
		for (int i = 0; i < neu[n].Poisson_input_num; i++)
		{
			if (neu[n].Poisson_input_time[i] > t1 && neu[n].Poisson_input_time[i] <= t + dt)
			{
				evolve_model_with_correct_timestep(n, a, t1, neu[n].Poisson_input_time[i] - t1);
				t1 = neu[n].Poisson_input_time[i];
				double v_start = a.v;
				a.v += n < NE ? f[0] : f[1];

				if (a.if_fired == 0 && v_start < V_th && a.v >= V_th && t - a.last_fire_time >= T_ref)
				{
					a.if_fired = 1;
					a.last_fire_time = t1;
				}
			}
		}
		evolve_model_with_correct_timestep(n, a, t1, t + dt - t1);
	}
}


void Exchange(struct neuron &a, struct neuron b)  // a <-- b
{
	a.t = b.t;
	a.Nu = b.Nu;
	a.v = b.v;
	a.dv = b.dv;
	a.w = b.w;
	a.I_input = b.I_input;
	
	a.last_fire_time = b.last_fire_time;
	a.if_fired = 0;
}


//-----------------------------------------------------------------------------
//		find next firing time in [t, t1] for single neuron,  if found, recorded by a_old 
//      (to find the first spike time. i.e. spike-spike corrections)
//-----------------------------------------------------------------------------

void find_next_fire_time_for_single_neu(int id, struct neuron a, struct neuron &a_old, double t, double t1)
{
	Exchange(a_old, a);
	a_old.if_fired = 0;

	Check_update_conti_or_Poisson_input(id, a_old, t, t1 - t);

}


//-----------------------------------------------------------------------------
//		Update all neurons from neu.t --> t and check if fires during [t, t1]
//-----------------------------------------------------------------------------

void update_all_neu(struct neuron *a, struct neuron *a_old, double t, double t1)
{
#pragma omp parallel for num_threads(num_threads_openmp)

	for (int id = 0; id < N; id++)
	{
		a[id].if_fired = 0;
		if ((a[id].wait_strength_E == 0) && (a[id].wait_strength_I == 0))  
			continue;

		Check_update_conti_or_Poisson_input(id, a[id], a[id].t, t - a[id].t);
	
		double v_start = a[id].v;

		a[id].v += a[id].wait_strength_E;
		a[id].v -= a[id].wait_strength_I;

		a[id].wait_strength_E = 0;
		a[id].wait_strength_I = 0;	


		//// check fire 
		if ( a[id].if_fired == 0 && v_start < V_th && a[id].v >= V_th && t - a[id].last_fire_time >= T_ref)
		{
			a[id].last_fire_time = t;
			a[id].if_fired = 1;
		}

		if (!a[id].if_fired)
			find_next_fire_time_for_single_neu(id, a[id], a_old[id], t, t1);
	}

	int continue_update = 0;
	for (int id = 0; id < N; id++)
	{
		if (a[id].if_fired)
		{
			continue_update = 1;
			if (record_data[0])
			{
				fwrite(&t, sizeof(double), 1, FP);
				double s = id;
				fwrite(&s, sizeof(double), 1, FP);
			}
			a[id].fire_num++;
			a[id].last_fire_time = t;

			for (int i = 0; i < N; i++)
			{
				if (i == id)
					continue;
				if (id < NE && i < NE)
					a[i].wait_strength_E += CS[id][i];
				else if (id < NE && i >= NE)
					a[i].wait_strength_E += CS[id][i];
				else if (id >= NE && i < NE)
					a[i].wait_strength_I += CS[id][i];
				else
					a[i].wait_strength_I += CS[id][i];
			}    
			find_next_fire_time_for_single_neu(id, a[id], a_old[id], t, t1);
		}
	}

	if(continue_update)
		update_all_neu(a, a_old, t, t1);
}

//-----------------------------------------------------------------------------
//		Event driven
//-----------------------------------------------------------------------------

void evolve_model_with_initial_timestep(struct neuron *a, struct neuron *a_old, double t, double dt)
{
#pragma omp parallel for num_threads(num_threads_openmp)

	for (int i = 0; i < N; i++)        // parallel
		find_next_fire_time_for_single_neu(i, a[i], a_old[i], t, t + dt);

	while (1)
	{
		double first_fire_time = t + dt;
		int first_fire_neu = -1;

		for (int i = 0; i < N; i++)
		{
			if (a_old[i].if_fired && a_old[i].last_fire_time < first_fire_time)
			{
				first_fire_time = a_old[i].last_fire_time;
				first_fire_neu = i; 
			}
		}

		if (first_fire_neu > -1)
		{
			int id = first_fire_neu;

			Check_update_conti_or_Poisson_input(id, a[id], a[id].t, first_fire_time - a[id].t);

			if (record_data[0])
			{
				fwrite(&first_fire_time, sizeof(double), 1, FP);
				double s = id;
				fwrite(&s, sizeof(double), 1, FP);
			}
			a[id].fire_num++;
			a[id].last_fire_time = first_fire_time;

			for (int i = 0; i < N; i++)
			{
				if (i == id)
					continue;
				if (id < NE && i < NE)
					a[i].wait_strength_E += CS[id][i];
				else if (id < NE && i >= NE)
					a[i].wait_strength_E += CS[id][i];
				else if (id >= NE && i < NE)
					a[i].wait_strength_I += CS[id][i];
				else
					a[i].wait_strength_I += CS[id][i];
			}
			find_next_fire_time_for_single_neu(id, a[id], a_old[id], first_fire_time, t + dt);
			update_all_neu(a, a_old, first_fire_time, t + dt);
		}
		else
		{
			for (int i = 0; i < N; i++)
			{
				Exchange(a[i], a_old[i]);
			}
			break;
		}
	}
}

void Generate_Poisson_times(struct neuron &a, double t, double dt)
{
	int k = a.Poisson_input_num;
	double t1;
	if (k == -1)
		t1 = -log(1 - Random(a.seed)) / a.Nu; 
	else
		t1 = a.Poisson_input_time[k];      
	k = 0;
	a.Poisson_input_time[k] = t1;
	while (t1 <= t + dt)
	{
		k++;
		t1 += -log(1 - Random(a.seed)) / a.Nu;    // the last one is larger than t+dt

		if (k + 1 > int(T_step*a.Nu * 2) + 5)
			a.Poisson_input_time = (double *)realloc(a.Poisson_input_time, (k + 15) * sizeof(double));
		a.Poisson_input_time[k] = t1;
	}
	a.Poisson_input_num = k;
}


void Run_model()
{
	num_threads_openmp = N >= 8 ? 8 : 4;

	double t = 0, tt = 0, tt_fftw = 0, t_lib = 0, t_fp = 0;
	double t_test = 0;

	while (t < T_Max)
	{
		if (Nu > Epsilon)
			for (int i = 0; i < N; i++)
				Generate_Poisson_times(neu[i], t, T_step);

		evolve_model_with_initial_timestep(neu, neu_old, t, T_step);
		t += T_step;

		if (record_data[1] && t > Record_v_start && t <= Record_v_end && t - tt >= 0.5 - 1e-8)
		{
			tt = t;
			fwrite(&t, sizeof(double), 1, FP1);
			for (int id = 0; id < N; id++)
				fwrite(&neu[id].v, sizeof(double), 1, FP1);
		}



	}
}
