//-----------------------------------------------------------------------------
//       G_sse, G_ssi & G_ff update analytically
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//		Runge Kutta_2 mehtod: update neuron during determinsteristic parts(i.e. no outside input S and F)
//  	start from time t, end with time t+dt. There should update twice,
//		so can be simplyfied by one function.
//   Modified Euler Method
//-----------------------------------------------------------------------------

void Update_RK2(int n, struct neuron &a, double t, double dt)
{
	void Update_once_RK4(struct neuron a, double *k, double *w, int n, double t, double h);
	double v_start = a.v, dv_start = a.dv;
	double I_s, m_s, h_s, n_s;

	double w[4], w0[4];  // v,m,h,n
	double k[2][4];      // v,m,h,n


	if (Lib_method)
	{
		I_s = a.I_input, m_s = a.m;
		h_s = a.h, n_s = a.n;
	}

	w[0] = a.v, w[1] = a.m, w[2] = a.h, w[3] = a.n;

	for (int i = 0; i < 4; i++)
		w0[i] = w[i];
	Update_once_RK4(a, k[0], w0, n, t, dt);            //1

	for (int i = 0; i < 4; i++)
		w0[i] = w[i] + k[0][i];
	Update_neu_G(n, a, t, dt);  //include a.t=t+dt
	Update_once_RK4(a, k[1], w0, n, t + dt, dt);   //2

	for (int i = 0; i < 4; i++)
		w[i] = w[i] + (k[0][i] + k[1][i]) / 2.0;

	a.v = w[0], a.m = w[1], a.h = w[2], a.n = w[3];
	a.t = t + dt;

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
		//	a.last_fire_time = Find_roots_bisection(t, t + dt, v_start, a.v, dv_start, a.dv); 
		//a.last_fire_time = cubic_hermite_real_root(t, t + dt, v_start, a.v, dv_start, a.dv, V_th);

		a.last_fire_time = t + (V_th - v_start) / (a.v - v_start)*dt;
		a.if_fired = 1;
		if (Lib_method)
		{
			a.I_input = I_s+(a.I_input-I_s)/dt*(a.last_fire_time-t); //记录放电时刻的I,m,h,n的信息   
			a.m = m_s + (a.m - m_s) / dt * (a.last_fire_time - t);
			a.h = h_s + (a.h - h_s) / dt * (a.last_fire_time - t);
			a.n = n_s + (a.n - n_s) / dt * (a.last_fire_time - t);
		}

	}
	if (abs(a.v) > 1e3)
	{
		//printf("\nError! Too large time step %0.6f in RK2\n", T_step);
		//printf("n=%d dt=%0.2e last=%f t=%f v=%0.2e dv=%0.2e state=%d\n\n", n, dt, a.last_fire_time, a.t, a.v, a.dv, a.state);
		//getchar();// system("pause");
		a.v = -65; 
		//exit(1);
	}

	Call_num++;
}

