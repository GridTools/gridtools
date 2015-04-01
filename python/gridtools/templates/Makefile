#
# path to GridTools and Boost libs
#
GRIDTOOLS ?= ${GRIDTOOLS_ROOT}
BOOST ?= ${BOOST_ROOT}

#
# compiler flags
# 
CC := g++
#CCFLAGS  := -O0 -g -fPIC -fopenmp -DSUPPRESS_MESSAGES -DFLOAT_PRECISION=8 -DCXX11_DISABLE
CCFLAGS   := -O3 -fPIC -fopenmp -DNDEBUG -DSUPPRESS_MESSAGES -DFLOAT_PRECISION=8 -DCXX11_DISABLE
LIBRARIES := -fopenmp -lc
INCLUDES  := -I. -I$(BOOST)/include -I$(GRIDTOOLS)/include -I$(GRIDTOOLS)/include/communication/high-level -isystem $(GRIDTOOLS)/fussion/include
GENCODE   := -shared -Wl,-soname,lib{{ stencil.name|lower }}.so
OBJS      := {{ stencil.name }}.o $(GRIDTOOLS)/src/gridtools.o 


# Target rules
all: build

build: {{ stencil.name|lower }}

{{ stencil.name|lower }}: $(OBJS)
	$(CC) $(GENCODE) -o {{ stencil.lib_file }} $+ $(LIBRARIES)

%.o: %.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

clean:
	rm -f {{ stencil.lib_file }}
	rm -f $(OBJS)
