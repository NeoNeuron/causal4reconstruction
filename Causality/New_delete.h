#pragma once

void Create_Matrix()
{
	z = new double *[N*N];
	for (int i = 0; i < N*N; i++)
		z[i] = new double[2 * m[0] * m[1]];
	for (int i = 0; i < N*N; i++)
		for (int j = 0; j < 2 * m[0] * m[1]; j++)
			z[i][j] = 0;


	TE = new double*[N], GC = new double*[N];
	DMI = new double*[N], NCC = new double*[N];
	TE_2 = new double*[N], DMI_2 = new double*[N], NCC_2 = new double*[N];
	
	for (int i = 0; i < N; i++)
	{
		TE[i] = new double[N], GC[i] = new double[N];
		DMI[i] = new double[N], NCC[i] = new double[N];
		TE_2[i] = new double[N], DMI_2[i] = new double[N], NCC_2[i] = new double[N];
	}


}


void Delete_Matrix()
{
	for (int i = 0; i < N*N; i++)
		delete[]z[i];
	delete[]z;

	for (int i = 0; i < N; i++)
	{
		delete TE[i], delete  GC[i];
		delete DMI[i], delete  NCC[i];
		delete TE_2[i], delete  DMI_2[i], delete NCC_2[i];
	}
		
	delete TE, delete  GC;
	delete DMI, delete  NCC;
	delete TE_2, delete  DMI_2, delete NCC_2;
}