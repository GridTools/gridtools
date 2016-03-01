#pragma once

#ifdef __CUDACC__

#define assert(x)                                                              \
    do {                                                                \
        if (!(x)) {                                                     \
            printf("%s:%d: Assertion failed: '%s'\n",          \
                    __FILE__, __LINE__, #x);                            \
            abort();                                                    \
        }                                                               \
    } while (0)
#else
  #include <cassert>
#endif
