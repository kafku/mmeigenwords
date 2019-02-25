CXX=g++
OPT = -O3
CXXFLAGS = -g $(OPT) -Wall -Wextra -std=c++11 -fPIC -shared -fopenmp -march=native
CXXFLAGS += -DEIGEN_DEFAULT_TO_ROW_MAJOR
CXXFLAGS += -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
CONDA_ENV_ROOT := $(shell conda env list | grep "*" | awk '{print $$3}')
EIGEN_ROOT = ./eigen
BOOST_LIBS = -lboost_python3 -lboost_numpy3
MKL_LIBS =
BLAS_LIBS = -lopenblas
LIBS = ${BLAS_LIBS}
INCLUDES := -I${CONDA_ENV_ROOT}/include -I${EIGEN_ROOT} $(shell python3-config --includes | cut -d ' ' -f1)
LDFLAGS := -Wl,-rpath,${CONDA_ENV_ROOT}/lib \
		-L${CONDA_ENV_ROOT}/lib ${LIBS} $(shell python3-config --ldflags)

TARGET=wgs.so

all: ${TARGET}
.PHONY:clean

%.so : %.cpp
	${CXX} -o $@ $< ${CXXFLAGS} ${INCLUDES} ${LDFLAGS}

clean:
	rm *.so
