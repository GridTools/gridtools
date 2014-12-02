#
# path to gridtools
#
GRIDTOOLS_HOME ?= ${HOME}/src/gridtools

#
# compiler flags
# 
CC := g++
CCFLAGS := -fPIC -std=c++11 -DBACKEND_BLOCK -DFLOAT_PRECISION=8 -fopenmp
LIBRARIES := -lc
INCLUDES  := -I. -I$(GRIDTOOLS_HOME)/include -I$(GRIDTOOLS_HOME)/include/communication/high-level -isystem $(GRIDTOOLS_HOME)/fussion/include
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
