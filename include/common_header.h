#ifndef __COMMON_HEADER_
#define __COMMON_HEADER_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstring>
#include <malloc.h>
#include <omp.h>
#include <algorithm>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

template <class T> 
void str2vec(std::string str, std::vector<T>& out) {
    std::stringstream ss(str);
    T tmpVal;
    while (ss >> tmpVal)
        out.push_back(tmpVal);
}

#define MIN(a,b)  ((a)<(b)?(a):(b))
#define MAX(a,b)  ((a)>(b)?(a):(b))

#endif // __COMMON_HEADER_