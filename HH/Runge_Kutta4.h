//-----------------------------------------------------------------------------
//       G_sse, G_ssi & G_ff update analytically
//-----------------------------------------------------------------------------

void Update_once_RK4(struct neuron a, double *k, double *w, int n, double t, double h)
{
	double I_input;
	if (I_CONST)
		I_input = I_const_input;
	else
	{
		I_input = -(a.G_f+a.G_se)*(w[0] - V_G_E);
		I_input -= a.G_si * (w[0] - V_G_I);
	}

	k[0] = -G_Na*w[1] * w[1] * w[1] * w[2] * (w[0] - E_Na)
		- G_K*w[3] * w[3] * w[3] * w[3] * (w[0] - E_K) - G_L*(w[0] - E_L) + I_input;

	k[0] = k[0] * h / C;											 // v
	k[1] = h*(alpha_m(w[0]) - w[1] * (alpha_m(w[0]) + beta_m(w[0])));  // m
	k[2] = h*(alpha_h(w[0]) - w[2] * (alpha_h(w[0]) + beta_h(w[0])));  // h
	k[3] = h*(alpha_n(w[0]) - w[3] * (alpha_n(w[0]) + beta_n(w[0])));  // n

}

//-----------------------------------------------------------------------------
//		Runge Kutta_4 mehtod: update neuron during determinsteristic parts(i.e. no outside input S and F)
//  	start from time t, end with time t+dt. There are four times to update,
//		so can be simplyfied by one function.
//
//		We just update v,m,h,n with RK4 method, since conductance parts are correct solutions.
//-----------------------------------------------------------------------------

void Update_RK4(int n, struct neuron &a, double t, double dt)
{
	void Update_neu_G(int n, struct neuron &a, double t, double dt);
	double v_start = a.v, dv_start = a.dv;
	double I_s, m_s, h_s, n_s;

	double w[4], w0[4];  // v,m,h,n
	double k[4][4];      // v,m,h,n

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
		w0[i] = w[i] + 0.5*k[0][i];	
	Update_neu_G(n, a, t, dt / 2); //include a.t=t+dt
	Update_once_RK4(a, k[1], w0, n, t + dt / 2, dt);   //2


	for (int i = 0; i < 4; i++)
		w0[i] = w[i] + 0.5*k[1][i];
	Update_once_RK4(a, k[2], w0, n, t + dt / 2, dt);   //3

	for (int i = 0; i < 4; i++)
		w0[i] = w[i] + k[2][i];
	Update_neu_G(n, a, t + dt / 2, dt / 2); //include a.t=t+dt
	Update_once_RK4(a, k[3], w0, n, t + dt, dt);   //4

	for (int i = 0; i < 4; i++)
		w[i] = w[i] + (k[0][i] + k[1][i] * 2 + k[2][i] * 2 + k[3][i]) / 6.0;
	

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
			a.I_input = I_s + (a.I_input - I_s) / dt * (a.last_fire_time - t); //记录放电时刻的I,m,h,n的信息   
			a.m = m_s + (a.m - m_s) / dt * (a.last_fire_time - t);
			a.h = h_s + (a.h - h_s) / dt * (a.last_fire_time - t);
			a.n = n_s + (a.n - n_s) / dt * (a.last_fire_time - t);
		}
	}

	if (abs(a.v) > 1e3)
	{
		printf("\nError! Too large time step %0.6f in RK4\n", T_step);
		printf("n=%d dt=%0.2e last=%f t=%f v=%0.2e dv=%0.2e state=%d\n\n", n, dt, a.last_fire_time, a.t, a.v, a.dv, a.state);
		getchar();// system("pause");
		//a.v = -65; a.dv = 0;
		exit(1);
	}
	Call_num++;
}


//-----------------------------------------------------------------------------
// Lib method, evolve to Tref. neu.t may be larger than t+dt, just unpdate G_s,G_ss
// G_f, G_ff
//-----------------------------------------------------------------------------

void Update_neu_G(int n, struct neuron &a, double t, double dt)
{
	double s_d_e, s_d_i, s_r_e, s_r_i;

	if (abs(dt - T_step) < 1e-8)
	{
		s_d_e = S_d_e, s_d_i = S_d_i;
		s_r_e = S_r_e, s_r_i = S_r_i;
	}
	else
	{
		s_d_e = exp(-dt / Sigma_d_E), s_d_i = exp(-dt / Sigma_d_I);
		s_r_e = exp(-dt / Sigma_r_E), s_r_i = exp(-dt / Sigma_r_I);
	}

	a.G_se = a.G_se * s_r_e + Sigma_ratio_E*a.G_sse * (s_d_e - s_r_e);
	a.G_sse = a.G_sse * s_d_e;
	a.G_si = a.G_si * s_r_i + Sigma_ratio_I*a.G_ssi * (s_d_i - s_r_i);
	a.G_ssi = a.G_ssi * s_d_i;
	if (Nu > Epsilon)
	{
		a.G_f = a.G_f * s_r_e + Sigma_ratio_E*a.G_ff * (s_d_e - s_r_e);
		a.G_ff = a.G_ff * s_d_e;
	}

	a.t = t + dt;	
}
