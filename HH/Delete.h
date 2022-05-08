void Delete()
{
	for (int i = 0; i < N; i++)
	{
		delete[] Connect_Matrix[i];
		delete[] CS[i];
		delete[] neu[i].Poisson_input_time;
	}
	delete[] Connect_Matrix;
	delete[] CS;
	delete[] neu;
	//, delete[] neu_old;

	if (record_data[0])
		fclose(FP); 
	if (record_data[1])
		fclose(FP1);
	
	if (Power_spectrum)
		fclose(FP_FFTW);
	if (RecordFP)
		fclose(FP_fire_pattern);

	if (Lib_method == 1)
	{
		for (int i = 0; i < 8; i++)
			delete[]Lib_data[i];
		delete[] Lib_data;

		for (int i = 0; i < 4; i++)
			delete[]Lib_unique[i];
		delete[]Lib_unique;
		delete[]Lib_resolution;

		if(Power_spectrum)
		{
			for (int i = 0; i < int(T_ref * 256 + 0.1); i++)
				delete[] Lib_v[i];
			delete[]Lib_v;
		}
	}

}