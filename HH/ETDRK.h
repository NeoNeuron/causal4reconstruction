#pragma once

void find_A(struct neuron &a, double *A, double *F)
{
	double s;

	//v
	//A[0] = -G_Na*a.m * a.m * a.m * a.h - G_K*a.n * a.n * a.n * a.n - G_L; ///revise
	A[0] = -G_Na * a.m * a.m * a.m * a.h - G_K * a.n * a.n * a.n * a.n - G_L - a.G_f - a.G_se - a.G_si;	
	F[0] = a.dv;

	//m
	s = alpha_m(a.v);
	A[1] = -(s + beta_m(a.v));
	F[1] = A[1] * a.m + s;

	//h
	s = alpha_h(a.v);
	A[2] = -(s + beta_h(a.v));
	F[2] = A[2] * a.h + s;

	//n
	s = alpha_n(a.v);
	A[3] = -(s + beta_n(a.v));
	F[3] = A[3] * a.n + s;
}

void Update_ETDRK4(int n, struct neuron &a, double t, double dt)
{
	double v_start = a.v, dv_start = a.dv;

	if (t >= a.last_fire_time + T_ref ) /// out side the stiff period
		Update_RK4(n, a, t, dt);
	else                                   /// inside the stiff period
	{
		double h = dt, A[4]; // v,m,h,n  A[]x    
		double g[4];
		double F[4][4], FF[4][4]; //F--dv,dm,dh,dn, FF=F-A*u
		double an[4], bn[4], cn[4]; //, an--v,m,h,n F1--a,2--b,3--c

		find_A(a, A, F[0]);

		for (int i = 0; i < 4; i++)
			g[i] = exp(A[i] * h / 2);

		FF[0][0] = F[0][0] - A[0] * a.v;
		FF[0][1] = F[0][1] - A[1] * a.m;
		FF[0][2] = F[0][2] - A[2] * a.h;
		FF[0][3] = F[0][3] - A[3] * a.n;

		an[0] = a.v*g[0] + FF[0][0] * (g[0] - 1) / A[0]; //v
		an[1] = a.m*g[1] + FF[0][1] * (g[1] - 1) / A[1]; //m
		an[2] = a.h*g[2] + FF[0][2] * (g[2] - 1) / A[2]; //h
		an[3] = a.n*g[3] + FF[0][3] * (g[3] - 1) / A[3]; //n
		///////////////////// 1

		Update_neu_G(n, a, t, h / 2); //include a.t=t+dt  半步长，可能有浮点运算问题
		double I_input;
		if (I_CONST)
			I_input = I_const_input;
		else
		{
			I_input = -(a.G_f + a.G_se)*(an[0] - V_G_E);
			I_input -= a.G_si * (an[0] - V_G_I);
		}

		F[1][0] = -G_Na*an[1] * an[1] * an[1] * an[2] * (an[0] - E_Na)
			- G_K*an[3] * an[3] * an[3] * an[3] * (an[0] - E_K) - G_L*(an[0] - E_L) + I_input;
		F[1][1] = alpha_m(an[0]) - an[1] * (alpha_m(an[0]) + beta_m(an[0]));  //m
		F[1][2] = alpha_h(an[0]) - an[2] * (alpha_h(an[0]) + beta_h(an[0]));  //h
		F[1][3] = alpha_n(an[0]) - an[3] * (alpha_n(an[0]) + beta_n(an[0]));  //n

		for (int i = 0; i < 4; i++)
			FF[1][i] = F[1][i] - A[i] * an[i];

		bn[0] = a.v*g[0] + FF[1][0] * (g[0] - 1) / A[0]; //v
		bn[1] = a.m*g[1] + FF[1][1] * (g[1] - 1) / A[1]; //m
		bn[2] = a.h*g[2] + FF[1][2] * (g[2] - 1) / A[2]; //h
		bn[3] = a.n*g[3] + FF[1][3] * (g[3] - 1) / A[3]; //n
		////////////////////// 2

		if (I_CONST)
			I_input = I_const_input;
		else
		{
			I_input = -(a.G_f + a.G_se)*(bn[0] - V_G_E);
			I_input -= a.G_si * (bn[0] - V_G_I);
		}

		F[2][0] = -G_Na*bn[1] * bn[1] * bn[1] * bn[2] * (bn[0] - E_Na)
			- G_K*bn[3] * bn[3] * bn[3] * bn[3] * (bn[0] - E_K) - G_L*(bn[0] - E_L) + I_input;
		F[2][1] = alpha_m(bn[0]) - bn[1] * (alpha_m(bn[0]) + beta_m(bn[0]));  //m
		F[2][2] = alpha_h(bn[0]) - bn[2] * (alpha_h(bn[0]) + beta_h(bn[0]));  //h
		F[2][3] = alpha_n(bn[0]) - bn[3] * (alpha_n(bn[0]) + beta_n(bn[0]));  //n

		for (int i = 0; i < 4; i++)
			FF[2][i] = F[2][i] - A[i] * bn[i];

		cn[0] = an[0] * g[0] + (2 * FF[2][0] - FF[0][0]) * (g[0] - 1) / A[0]; //v
		cn[1] = an[1] * g[1] + (2 * FF[2][1] - FF[0][1]) * (g[1] - 1) / A[1]; //m
		cn[2] = an[2] * g[2] + (2 * FF[2][2] - FF[0][2]) * (g[2] - 1) / A[2]; //h
		cn[3] = an[3] * g[3] + (2 * FF[2][3] - FF[0][3]) * (g[3] - 1) / A[3]; //n
		/////////////////////  3


		Update_neu_G(n, a, t + h / 2, h / 2); //include a.t=t+dt  半步长，可能有浮点运算问题   
		if (I_CONST)
			I_input = I_const_input;
		else
		{
			I_input = -(a.G_f + a.G_se)*(cn[0] - V_G_E);
			I_input -= a.G_si * (cn[0] - V_G_I);
		}

		F[3][0] = -G_Na*cn[1] * cn[1] * cn[1] * cn[2] * (cn[0] - E_Na)
			- G_K*cn[3] * cn[3] * cn[3] * cn[3] * (cn[0] - E_K) - G_L*(cn[0] - E_L) + I_input;
		F[3][1] = alpha_m(cn[0]) - cn[1] * (alpha_m(cn[0]) + beta_m(cn[0]));  //m
		F[3][2] = alpha_h(cn[0]) - cn[2] * (alpha_h(cn[0]) + beta_h(cn[0]));  //h
		F[3][3] = alpha_n(cn[0]) - cn[3] * (alpha_n(cn[0]) + beta_n(cn[0]));  //n

		for (int i = 0; i < 4; i++)
			FF[3][i] = F[3][i] - A[i] * cn[i];
		////////////////////   4

		for (int i = 0; i < 4; i++)
			g[i] = g[i] * g[i];     //exp(A*h)

		double gm[4][3] = {0}, s[4];
		double c0[6] = { 5 / 1008.0, 1 / 45.0, 3 / 40.0, 1 / 6.0, 1 / 6.0, 0 };
		double c1[6] = { 1.0 / 504, 1.0 / 90, 1.0 / 20, 1.0 / 6, 1.0 / 3,  0 };
		double c2[6] = { -1.0 / 1680, -1.0 / 360, -1.0 / 120, 0, 1.0 / 6,  0 };
		for (int i = 0; i < 4; i++)
		{
			s[i] = A[i] * h;
			if (abs(s[i]) >= 1e-2) ///good
			{
				gm[i][0] = (-4 - s[i] + g[i] * (4 - 3 * s[i] + s[i] * s[i])) / s[i] / s[i] / A[i];
				gm[i][1] = 2 * (2 + s[i] + g[i] * (-2 + s[i])) / s[i] / s[i] / A[i];
				gm[i][2] = (-4 - 3 * s[i] - s[i] * s[i] + g[i] * (4 - s[i])) /s[i] / s[i] / A[i];
			}
			else
			{
				for (int j = 1; j < 6; j++)     //Series[, {s,0,4}] , O(s^4)
					gm[i][0] = gm[i][0] * s[i] + c0[j];
				gm[i][0] /= A[i];

				for (int j = 1; j < 6; j++)
					gm[i][1] = gm[i][1] * s[i] + c1[j];
				gm[i][1] /= A[i];

				for (int j = 1; j < 6; j++)
					gm[i][2] = gm[i][2] * s[i] + c2[j];
				gm[i][2] /= A[i];

			}
		}

		a.v = a.v*g[0] + FF[0][0] * gm[0][0] + (FF[1][0] + FF[2][0])*gm[0][1] + FF[3][0] * gm[0][2];
		a.m = a.m*g[1] + FF[0][1] * gm[1][0] + (FF[1][1] + FF[2][1])*gm[1][1] + FF[3][1] * gm[1][2];
		a.h = a.h*g[2] + FF[0][2] * gm[2][0] + (FF[1][2] + FF[2][2])*gm[2][1] + FF[3][2] * gm[2][2];
		a.n = a.n*g[3] + FF[0][3] * gm[3][0] + (FF[1][3] + FF[2][3])*gm[3][1] + FF[3][3] * gm[3][2];
		a.t = t + dt;      // 半步长，可能有浮点运算问题, 重新计算

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

			printf("Warning! in the stiff period and fire ETDRK\n");			///edit
			printf("n=%d v1=%0.2f v2=%0.2f last=%0.3f t=%0.3f dt=%0.2e state=%d\n\n",n,v_start,a.v ,a.last_fire_time,t,dt,a.state); 
		}

		if (abs(a.v) > 1e3)
		{
			printf("\nError! Too large time step %0.6f in ETDRK4\n", T_step);
			printf("n=%d dt=%0.2e last=%f t=%f v=%0.2e dv=%0.2e state=%d\n\n", n, dt, a.last_fire_time, a.t, a.v, a.dv, a.state);

			for (int i = 0; i < 4; i++)
			{
				printf("i=%d A=%0.3e h=%0.3e %e\n", i, A[i], h, A[i] * h);
			}

			//getchar();// system("pause");
			//a.v = -65; a.dv = 0;
			exit(1);
		}
		Call_num++;

	}

}

void Update_ETDRK2(int n, struct neuron &a, double t, double dt)
{
	double v_start = a.v, dv_start = a.dv;


	if (t >= a.last_fire_time + T_ref)       /// out side the stiff period
		Update_RK2(n, a, t, dt);
	else									 /// inside the stiff period
	{
		double h = dt, A[4]; // v,m,h,n  A[]x    
		double g[4];
		double F[2][4], FF[2][4]; //F--dv,dm,dh,dn, FF=F-A*u
		double an[4]; //an--v,m,h,n 

		find_A(a, A, F[0]);

		for (int i = 0; i < 4; i++)
			g[i] = exp(A[i] * h);

		FF[0][0] = F[0][0] - A[0] * a.v;
		FF[0][1] = F[0][1] - A[1] * a.m;
		FF[0][2] = F[0][2] - A[2] * a.h;
		FF[0][3] = F[0][3] - A[3] * a.n;

		an[0] = a.v*g[0] + FF[0][0] * (g[0] - 1) / A[0]; //v
		an[1] = a.m*g[1] + FF[0][1] * (g[1] - 1) / A[1]; //m
		an[2] = a.h*g[2] + FF[0][2] * (g[2] - 1) / A[2]; //h
		an[3] = a.n*g[3] + FF[0][3] * (g[3] - 1) / A[3]; //n
		///////////////////// 1

		Update_neu_G(n, a, t, h); //include a.t=t+dt

		double I_input;
		if (I_CONST)
			I_input = I_const_input;
		else
		{
			I_input = -(a.G_f + a.G_se)*(an[0] - V_G_E);
			I_input -= a.G_si * (an[0] - V_G_I);
		}
		F[1][0] = -G_Na*an[1] * an[1] * an[1] * an[2] * (an[0] - E_Na)
			- G_K*an[3] * an[3] * an[3] * an[3] * (an[0] - E_K) - G_L*(an[0] - E_L) + I_input;
		F[1][1] = alpha_m(an[0]) - an[1] * (alpha_m(an[0]) + beta_m(an[0]));  //m
		F[1][2] = alpha_h(an[0]) - an[2] * (alpha_h(an[0]) + beta_h(an[0]));  //h
		F[1][3] = alpha_n(an[0]) - an[3] * (alpha_n(an[0]) + beta_n(an[0]));  //n

		for (int i = 0; i < 4; i++)
			FF[1][i] = F[1][i] - A[i] * an[i];
		////////////////////// 2
		double gm[4][2] = {0}, s[4];


		double c0[6] = { 1.0 / 144, 1.0 / 30, 1.0 / 8,1.0 / 3,1.0 / 2,0 };
		double c1[6] = { 1.0 / 720,1.0 / 120,1.0 / 24,1.0 / 6,1.0 / 2,0 };

		for (int i = 0; i < 4; i++)
		{
			s[i] = A[i] * h;
			if (abs(s[i]) >= 1e-3) ///good
			{
				gm[i][0] = (1 + g[i] * (s[i] - 1)) / s[i] / A[i];  
				gm[i][1] = (g[i] - 1 - s[i]) / s[i] / A[i];        
			}
			else
			{
				for (int j = 1; j < 6; j++)     //Series[, {s,0,4}] , O(s^4)
					gm[i][0] = gm[i][0] * s[i] + c0[j];
				gm[i][0] /= A[i];

				for (int j = 1; j < 6; j++)
					gm[i][1] = gm[i][1] * s[i] + c1[j];
				gm[i][1] /= A[i];
			}

		}

		a.v = a.v*g[0] + FF[0][0] * gm[0][0] + FF[1][0] * gm[0][1];
		a.m = a.m*g[1] + FF[0][1] * gm[1][0] + FF[1][1] * gm[1][1];
		a.h = a.h*g[2] + FF[0][2] * gm[2][0] + FF[1][2] * gm[2][1];
		a.n = a.n*g[3] + FF[0][3] * gm[3][0] + FF[1][3] * gm[3][1];

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
			//printf("Warning! in the stiff period and fire ETDRK\n");
			//printf("n=%d v1=%0.2f v2=%0.2f last=%0.3f t=%0.3f dt=%0.2e state=%d\n\n", n, v_start, a.v, a.last_fire_time, t, dt, a.state);
		}

		if (abs(a.v) > 1e3)
		{
			//printf("\nError! Too large time step %0.6f IN ETDRK2\n", T_step);
			//printf("n=%d dt=%0.2e last=%f t=%f v=%0.2e dv=%0.2e\n\n", n, dt, a.last_fire_time, a.t, a.v, a.dv);
			//getchar();// system("pause");
			a.v = -65; 
			//exit(1);
		}
		Call_num++;
	}
}




