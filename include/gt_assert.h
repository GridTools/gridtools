#pragma once

#ifdef __CUDACC__
#undef assert
#define assert(e)
#else
#include <cassert>
#endif

