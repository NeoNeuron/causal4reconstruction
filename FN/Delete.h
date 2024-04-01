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
	delete[] neu, delete[] neu_old;

	if (record_data[0])
		fclose(FP); 
	if (record_data[1])
		fclose(FP1);
	
}