#
# path to gridtools
#
GRIDTOOLS_HOME ?= ${HOME}/src/gridtools

#
# compiler flags
# 
CC := c++
CCFLAGS := -fPIC -std=c++11 -DBACKEND_BLOCK -DFLOAT_PRECISION=8 -DCXX11_DISABLE -fopenmp
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
	$(CC) $(GENCODE) -o lib$@.so $+ $(LIBRARIES)

{{ stencil.cpp_file }}.o: {{ stencil.cpp_file }}
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

clean:
	rm -f {{ stencil.cpp_file }}
