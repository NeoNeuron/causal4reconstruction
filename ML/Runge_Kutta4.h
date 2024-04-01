//-----------------------------------------------------------------------------
//       G_sse, G_ssi & G_ff update analytically
//-----------------------------------------------------------------------------

void Update_once_RK4(double *k, double *w, int n, double t, double h)
{
	double Mss, nss, tau_n, I_input;

	if (I_CONST)
		I_input = I_const_input;
	else
		I_input = 0;

	Mss = 0.5*(1 + tanh((w[0] - V1) / V2));
	nss = 0.5*(1 + tanh((w[0] - V3) / V4));
	tau_n =  phi * cosh((w[0] - V3) / 2.0 / V4);


	k[0] = -G_L * (w[0] - E_L) - G_Ca * Mss*(w[0] - E_Ca) - G_K * w[1] * (w[0] - E_K) + I_input;
	k[0] = k[0] * h / C;

	k[1] = h * (nss - w[1]) * tau_n;


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
	double v_start = a.v, dv_start = a.dv;

	double w[2], w0[2];  // v,n
	double k[4][2];      // v,n


	w[0] = a.v, w[1] = a.n;


	for (int i = 0; i < 2; i++)
		w0[i] = w[i];
	Update_once_RK4(k[0], w0, n, t, dt);            //1	

	for (int i = 0; i < 2; i++)
		w0[i] = w[i] + 0.5*k[0][i];
	Update_once_RK4(k[1], w0, n, t + dt / 2, dt);   //2


	for (int i = 0; i < 2; i++)
		w0[i] = w[i] + 0.5*k[1][i];
	Update_once_RK4(k[2], w0, n, t + dt / 2, dt);   //3


	for (int i = 0; i < 2; i++)
		w0[i] = w[i] + k[2][i];
	Update_once_RK4(k[3], w0, n, t + dt, dt);   //4

	for (int i = 0; i < 2; i++)
		w[i] = w[i] + (k[0][i] + k[1][i] * 2 + k[2][i] * 2 + k[3][i]) / 6.0;

	a.v = w[0], a.n = w[1];
	a.t = t + dt;

	double Mss;
	Mss = 0.5*(1 + tanh((a.v - V1) / V2));

	if (I_CONST)
		a.I_input = I_const_input;
	else
		a.I_input = 0;
		
	a.dv = G_L * (a.v - E_L) - G_Ca * Mss*(a.v - E_Ca) - G_K * a.n * (a.v - E_K) + a.I_input;
	a.dv /= C;

	if (v_start < V_th && a.v >= V_th && t + dt - a.last_fire_time >= T_ref)
	{
		a.last_fire_time = cubic_hermite_real_root(t, t + dt, v_start, a.v, dv_start, a.dv, V_th);
		a.if_fired = 1;
	}

	if (abs(a.v) > 1e3)
	{
		printf("\nError! Too large time step %0.6f in RK4\n", T_step);
		printf("n=%d dt=%0.2e last=%f t=%f v=%0.2e dv=%0.2e state=%d\n\n", n, dt, a.last_fire_time, a.t, a.v, a.dv, a.state);
		getchar();// system("pause");
		exit(1);
	}
}

