# define compiler and path of libs
CPPFLAGS = --std=c++11 -w -I $(DIR_INC) -I HH
CXXFLAGS = -fopenmp -g -O2 
LDLIBS = -lboost_program_options -fopenmp
# define variable path
DIR_INC = include
DIR_SRC = HH Causality HHcon Lorenz Lcon Logistic FN ML 
DIR_BIN = bin

vpath %.cpp $(DIR_SRC)
vpath %.h   $(DIR_INC)
vpath %.h   $(DIR_SRC)

HEADERS_COMMON := $(notdir $(wildcard $(DIR_INC)/*.h))
HEADERS_HH := $(notdir $(wildcard HH/*.h))
HEADERS_Causality := $(notdir $(wildcard Causality/*.h))
SRCS_Causality = $(wildcard Causality/*.cpp)
OBJS_Causality = $(SRCS_Causality:.cpp=.o)
HEADERS_HHcon := $(notdir $(wildcard HHcon/*.h))
HEADERS_Lorenz := $(notdir $(wildcard Lorenz/*.h))
HEADERS_Lcon := $(notdir $(wildcard Lcon/*.h))
BIN := $(DIR_BIN)/calCausality $(DIR_BIN)/simHH $(DIR_BIN)/simHHcon $(DIR_BIN)/simLorenz $(DIR_BIN)/simLcon $(DIR_BIN)/simFN $(DIR_BIN)/simML $(DIR_BIN)/simLogistic

.PHONY : all
all : $(BIN)

$(DIR_BIN)/calCausality : $(OBJS_Causality) $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Causality)
	$(CXX) $(CPPFLAGS) -o $(DIR_BIN)/calCausality $(OBJS_Causality) $(LDLIBS)

$(DIR_BIN)/simHH : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_HH) 
	$(CXX) $(CPPFLAGS) -O2 HH/main.cpp -o $(DIR_BIN)/simHH $(LDLIBS)

$(DIR_BIN)/simHHcon : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_HHcon) 
	$(CXX) $(CPPFLAGS) HHcon/main.cpp -o $(DIR_BIN)/simHHcon $(LDLIBS)

$(DIR_BIN)/simLorenz : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Lorenz) 
	$(CXX) $(CPPFLAGS) Lorenz/main.cpp -o $(DIR_BIN)/simLorenz $(LDLIBS)

$(DIR_BIN)/simLcon : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Lcon) 
	$(CXX) $(CPPFLAGS) Lcon/main.cpp -o $(DIR_BIN)/simLcon $(LDLIBS)

$(DIR_BIN)/simFN : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_FN) 
	$(CXX) $(CPPFLAGS) FN/main.cpp -o $(DIR_BIN)/simFN $(LDLIBS)

$(DIR_BIN)/simML : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_ML) 
	$(CXX) $(CPPFLAGS) ML/main.cpp -o $(DIR_BIN)/simML $(LDLIBS)

$(DIR_BIN)/simLogistic : $(DIR_BIN) $(HEADERS_COMMON) $(HEADERS_Logistic) 
	$(CXX) $(CPPFLAGS) Logistic/main.cpp -o $(DIR_BIN)/simLogistic $(LDLIBS)

.PHONY : debug
debug : CPPFLAGS += -DDEBUG
debug : $(BIN)

$(DIR_BIN) : 
	@mkdir -p $@

.PHONY : clean
clean:
	rm -rf $(DIR_BIN)
