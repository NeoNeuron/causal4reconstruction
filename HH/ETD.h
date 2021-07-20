#pragma once
void Update_F(struct neuron &a)
{
	int id = (a.id_F + 1) % ode_type;
	a.F[id][0] = a.t;
	a.F[id][1] = a.v;
	a.F[id][2] = a.m;
	a.F[id][3] = a.h;
	a.F[id][4] = a.n;

	a.F[id][5] = a.dv;
	a.F[id][6] = alpha_m(a.v) - a.m * (alpha_m(a.v) + beta_m(a.v));
	a.F[id][7] = alpha_h(a.v) - a.h * (alpha_h(a.v) + beta_h(a.v));
	a.F[id][8] = alpha_n(a.v) - a.n * (alpha_n(a.v) + beta_n(a.v));

	a.id_F = id;       ////each id_F stands for the latest information	
}
	
double ETD2_ToFind_vmhn(struct neuron &a, int ID, double h, double h1)
{
	/// ID 1--v,2--m,3--h,4--n
	double gm[3];
	double A;
	int id, id1;
	id = a.id_F, id1 = (id + ode_type - 1) % ode_type;

	if (ID == 1) // v
		A = -G_Na*a.m * a.m * a.m * a.h - G_K*a.n * a.n * a.n * a.n - G_L;
	else if (ID == 2) // m
		A = (a.F[id][ID + 4] - alpha_m(a.v)) / a.F[id][ID];
	else if (ID == 3)  //h
		A = (a.F[id][ID + 4] - alpha_h(a.v)) / a.F[id][ID];
	else if (ID == 4)  //n
		A = (a.F[id][ID + 4] - alpha_n(a.v)) / a.F[id][ID];
	else
	{
		printf("Error! wrong ETD2_ToFind_vmhn ID=%d\n", ID);
		exit(0);
	}


	gm[0] = exp(A*h);
	gm[1] = -1 - A*h1 - A*h + gm[0] * (1 + A*h1);
	gm[2] = 1 - gm[0] + A*h;

	return a.F[id][ID] * gm[0] + ((a.F[id][ID + 4] - A*a.F[id][ID])*gm[1] + (a.F[id1][ID + 4] - A*a.F[id1][ID])*gm[2]) / A / A / h1;

}

void Update_ETD2(int n, struct neuron &a, double t, double dt)
{
	double v_start = a.v, dv_start = a.dv;


	//double check_t;
	//check_t = (int((t - a.last_fire_time) / T_Step_Small) + 1)*T_Step_Small;
	
	

	if (t <= 3 * T_step)
	{
		Update_RK2(n, a, t, dt);
		Update_F(a);
		return ;
	}


	if (t + dt >= a.last_fire_time + T_ref) /// out side the stiff period
	{
		Update_RK2(n, a, t, dt);
		Update_F(a);
	}
	else              /// inside the stiff period
	{
		double h, h1;      /// h=t_n+1-t_n, h1=t_n-t_n-1
		double A[4];       ///v,m,h,n A[]x
		double gm[4][3];
		int id, id1;
		//double d[4];       /// denominator

		id = a.id_F, id1 = (id + ode_type - 1) % ode_type;
		h = dt;
		h1 = a.F[id][0] - a.F[id1][0];
		
		if (h < 1e-6) /// check divide 0
		{
			Update_RK2(n, a, t, dt);
			return;
		}


		A[0] = -G_Na*a.m * a.m * a.m * a.h - G_K*a.n * a.n * a.n * a.n - G_L;
		A[1] = (a.F[id][6] - alpha_m(a.v)) / a.F[id][2];
		A[2] = (a.F[id][7] - alpha_h(a.v)) / a.F[id][3];
		A[3] = (a.F[id][8] - alpha_n(a.v)) / a.F[id][4];


		for (int i = 0; i < 4; i++)
			gm[i][0] = exp(A[i] * h);
		for (int i = 0; i < 4; i++)
			gm[i][1] = (-1 - A[i] * h1 - A[i] * h + gm[i][0] * (1 + A[i] * h1)) / h1/A[i] / A[i];
		for (int i = 0; i < 4; i++)
			gm[i][2] = (1 - gm[i][0] + A[i] * h) / h1/A[i] / A[i];


		double s[5];
		s[0] = ETD2_ToFind_vmhn(a, 1, h, h1);
		s[1] = ETD2_ToFind_vmhn(a, 2, h, h1);
		s[2] = ETD2_ToFind_vmhn(a, 3, h, h1);
		s[3] = ETD2_ToFind_vmhn(a, 4, h, h1);

		a.v = a.F[id][1] * gm[0][0] + (a.F[id][5] - A[0] * a.F[id][1])*gm[0][1] + (a.F[id1][5] - A[0] * a.F[id1][1])*gm[0][2];
		a.m = a.F[id][2] * gm[1][0] + (a.F[id][6] - A[1] * a.F[id][2])*gm[1][1] + (a.F[id1][6] - A[1] * a.F[id1][2])*gm[1][2];
		a.h = a.F[id][3] * gm[2][0] + (a.F[id][7] - A[2] * a.F[id][3])*gm[2][1] + (a.F[id1][7] - A[2] * a.F[id1][3])*gm[2][2];
		a.n = a.F[id][4] * gm[3][0] + (a.F[id][8] - A[3] * a.F[id][4])*gm[3][1] + (a.F[id1][8] - A[3] * a.F[id1][4])*gm[3][2];


		Update_neu_G(n, a, t, dt);  //include a.t=t+dt

		if (I_CONST)
			a.I_input = I_const_input;
		else
		{
			a.I_input = -(a.G_f + a.G_se)*(a.v - V_G_E);
			a.I_input -= a.G_si * (a.v - V_G_I);
		}
		a.dv = -G_Na*a.m * a.m * a.m * a.h * (a.v - E_Na)
			- G_K*a.n * a.n * a.n * a.n * (a.v - E_K) - G_L*(a.v - E_L) + a.I_input;
		a.dv /= C;

		if (v_start < V_th && a.v >= V_th && t + dt - a.last_fire_time >= T_ref)
		{
			//a.last_fire_time = cubic_hermite_real_root(t, t + dt, v_start, a.v, dv_start, a.dv, V_th);

			a.last_fire_time = t + (V_th - v_start) / (a.v - v_start)*dt;
			a.if_fired = 1;
		}

		if (abs(a.v) > 1e3)
		{
			printf("\nError! Too large time step %0.6f in ETD2\n", T_step);
			printf("n=%d dt=%0.2e last=%f t=%f v=%0.2e dv=%0.2e\n\n", n, dt, a.last_fire_time, a.t, a.v, a.dv);
			//getchar();// system("pause");
			a.v = -65; 
			//exit(1);
		}
		Call_num++;
		Update_F(a);
	}
}

void Update_ETD4(int n, struct neuron &a, double t, double dt)
{
	printf("No ETD4 method!\n");
	exit(0);
}



