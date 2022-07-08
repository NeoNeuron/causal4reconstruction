double total_call_RK4_0(double R)
{
	double num, num0;
	double T, dt, dts;
	double p, p1;

	dt = T_Step_Large;
	dts = T_Step_Small;
	T = T_Max;
	p = T_ref*R;
	if (count_num == 0)
		p1 = 0;
	else
		p1 = syn_num / count_num;

//	printf("syn_num=%0.0f count=%0.0f\n", syn_num, count_num);
//	printf("p=%f p1=%f\n", p, p1);
	if (Lib_method)
	{
		num = (1 - p)*(Nu*dt + 1)*T_Max / dt;
		num += (1 - p1)*(2.0*(N-1)*R*T_Max + (N-1)*R*T_Max*dt*Nu / 2.0);
		num += 2*R*T_Max;
		num *= N;
	}
	else if (Adaptive_method)
	{
		int double_count = 0;
		if (dt / T_Step_Small - int(dt / T_Step_Small) < Epsilon)
			double_count = 1;
	//	printf("double_c = %d\n",double_count);

		//num = (Nu*dt + 1)*T_Max / dt+T_Max*R*(int(T_ref/T_Step_Small-1e-6+1)-int(T_ref/dt-1e-6)*double_count);
		//num += 2.0*N*R*T_Max + p1*N*R*T_Max*(Nu*dt / 2)\
		//	+ p1*(N-1)*R*T_Max*(int(dt / T_Step_Small)/2.0-0.5)+ R*T_Max*(int(dt / T_Step_Small) / 2.0 - 0.5);

		num = (Nu*dt + 1)*T_Max / dt + T_Max*R*(int(dt/T_Step_Small-1e-6)*(T_ref / dt-1) + 1 + dt/T_Step_Small);
		num += 2.0*N*R*T_Max + (N - 1)*R*T_Max*(Nu*dt / 2.0 + p1*dt / T_Step_Small / 2.0)\
			+ R*T_Max*Nu*dt / 2.0;

		num *= N;
	}
	else
	{
		num = (Nu*dt + 1)*T_Max / dt;
		num += (2.0*N*R*T_Max + N*R*T_Max*dt*Nu / 2.0);
		num *= N;
	}
	return num;
}

double total_call_RK4_1(double R)
{
	double num, dt, T, dts;
	double p, p1;

	dt = T_Step_Large;
	dts = T_Step_Small;
	p = T_ref*R;
	T = T_Max;

	if (count_num == 0)
		p1 = 0;
	else
		p1 = syn_num / count_num;

	if (Lib_method)
	{
		num = (1 - p)*(T / dt + T*Nu) + T*R*(2 + dt / 2 * Nu) +\
			P_c*(N - 1)*T*R*(2 + dt / 2 * Nu)*(1 - p1);
	}
	else if (Adaptive_method)
	{
		num = T / dt + T*Nu + T*R*(2 + dt / 2 * Nu + T_ref / dts + 1) + \
			P_c*(N - 1)*T*R*(2 + dt / 2 * Nu + p1*dt / 2 / dts);
	}
	else
	{
		num = T / dt + T*Nu + T*R*(2 + dt / 2 * Nu) + P_c*(N - 1)*T*R*(2 + dt / 2 * Nu);
	}
	return num*N;
}

double total_call_RK4_2(double R)
{
	double num, dt, T, dts;
	double p, p1;

	dt = T_Step_Large;
	dts = T_Step_Small;
	p = T_ref*R;
	T = T_Max;

	if (count_num == 0)
		p1 = 0;
	else
		p1 = syn_num / count_num;

	if (Lib_method)
	{
		num = (1 - p)*(T / dt + T*Nu) + T*R*(2 + dt  * Nu) + \
			P_c*(N - 1)*T*R*(2 + dt  * Nu)*(1 - p1);
	}
	else if (Adaptive_method)
	{
		num = T / dt + T*Nu + T*R*(2 + dt  * Nu + T_ref / dts + 1) + \
			P_c*(N - 1)*T*R*(2 + dt  * Nu + p1*dt / 2 / dts);
	}
	else
	{
		num = T / dt + T*Nu + T*R*(2 + dt  * Nu) + P_c*(N - 1)*T*R*(2 + dt  * Nu);
	}
	return num*N;
}

double new_total_call_RK4(double R)
{
	double num, dt, T, dts;
	double p, p1;

	dt = T_Step_Large;
	dts = T_Step_Small;
	p = T_ref*R;
	T = T_Max;

	if (count_num == 0)
		p1 = 0;
	else
		p1 = syn_num / count_num;

	if (Lib_method)
	{
		num = (1 - p)*(T / dt + T*Nu) + T*(2 + Nu*dt)*(R + R*P_c*(N - 1)*(1 - p1));
	}
	else if (Adaptive_method || ETD_method)
	{
		num = T / dt + T*Nu + T*(2 + Nu*dt)*(R + R*P_c*(N - 1)) + T*R*(T_ref / dts + 1) + T*R*P_c*(N - 1)*p1*dt / 2 / dts;
	}
	else
	{
		num = T / dt + T*Nu + T*(2 + Nu*dt)*(R + R*P_c*(N - 1)); 
	}
	return num*N;
}