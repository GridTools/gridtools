#
# path to gridtools and Boost libs
#
GRIDTOOLS ?= ${GRIDTOOLS_HOME}
BOOST ?= ${BOOST_ROOT}

#
# compiler flags
# 
CC := g++
#CCFLAGS := -O0 -g -fPIC -fopenmp -DNDEBUG -DFLOAT_PRECISION=8 -DCXX11_DISABLE
CCFLAGS := -O3 -fPIC -fopenmp -std=c++11 -DNDEBUG -DFLOAT_PRECISION=8 -DCXX11_DISABLE
LIBRARIES := -lc
INCLUDES  := -I. -I$(BOOST)/include -I$(GRIDTOOLS)/include -I$(GRIDTOOLS)/include/communication/high-level -isystem $(GRIDTOOLS)/fussion/include
GENCODE := -shared -Wl,-soname,lib{{ stencil.name|lower }}.so

#ifeq ($(CXX),c++)
#CCFLAGS+=-DGPU_LOCALITY
#INCLUDES+= -I$(HWLOC_PATH)/include
#LIBRARIES+= -L$(HWLOC_PATH)/lib -lhwloc -lcuda
#endif

# Target rules
all: build

build: {{ stencil.name|lower }}

{{ stencil.name|lower }}: {{ stencil.cpp_file }}.o
	$(CC) $(GENCODE) -o {{ stencil.lib_file }} $+ $(LIBRARIES)

{{ stencil.cpp_file }}.o: {{ stencil.cpp_file }}
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

clean:
	rm -f {{ stencil.lib_file }}
	rm -f {{ stencil.cpp_file }}.o
