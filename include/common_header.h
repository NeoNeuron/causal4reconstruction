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
#include <fstream>
#include <random>

// Fast code for exp(x) in double precision
#include "fmath.hpp"
//#define exp(x) fmath::expd(x)

template<typename Ty>
inline Ty my_expd(const Ty &x)
{ return exp(x); }

template<>
inline double my_expd(const double &x)
{ return fmath::expd(x); }

#define exp(x) my_expd(x)

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

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define WARNING(X) printf(ANSI_COLOR_YELLOW X ANSI_COLOR_RESET)
#define ERROR(X) printf(ANSI_COLOR_RED X ANSI_COLOR_RESET)

extern std::random_device rd;
extern std::mt19937 rng;

#endif // __COMMON_HEADER_