#pragma once
/**
@file
@brief global definitions
*/

#define GT_MAX_ARGS 8
#define GT_MAX_INDEPENDENT 3


#define GT_NO_ERRORS 0
#define GT_ERROR_NO_TEMPS 1

#if __cplusplus > 199711L
#ifndef CXX11_DISABLED
#define CXX11_ENABLED
#endif
#define CXX11_DISABLED
#endif

namespace gridtools{  namespace enumtype{
/** enum specifying the type of backend we use */
        enum backend  {Cuda, Host};

/** struct in order to perform templated methods partial specialization (Alexantrescu's trick, pre-c++1)*/
        template<backend T>
        struct backend_type
        {
            enum {value=T};
        };

        enum strategy  {Naive, Block};

/** struct in order to perform templated methods partial specialization (Alexantrescu's trick, pre-c++1)*/
        template<strategy T>
        struct strategy_type
        {
            enum {value=T};
        };
}
#ifndef CXX11_ENABLED
#define constexpr
#endif

#ifndef FLOAT_PRECISION
#define FLOAT_PRECISION 8
#endif

#if FLOAT_PRECISION == 4
    typedef float float_type;
#elif FLOAT_PRECISION == 8
    typedef double float_type;
#else
#error float precision not properly set (4 or 8 bytes supported)
#endif

}
