#include "common/defs.h"

extern "C"
int
getFloatSize() {
	return sizeof(gridtools::float_type)*8;
}
