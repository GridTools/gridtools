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

#include <boost/mpl/integral_c.hpp>
#ifdef CXX11_ENABLED
    //note: if we want to use in the same executable two stencils with different types for the ints
    //(e.g. to index a very large stencil), then we can template the following type definitions and
    //propagate the template everywhere! (which is one of the main motivations for the EPetra replacement in Trilinos)
    using int_t          =int ;
    using short_t        =int;
    using uint_t         =long unsigned int;
    using ushort_t       =unsigned  char;
    template<int_t N>
    using  static_int=boost::mpl::integral_c<int_t,N>;
    template<uint_t N>
    using  static_uint=boost::mpl::integral_c<uint_t,N>;
    template<short_t N>
    using  static_short=boost::mpl::integral_c<short_t,N>;
    template<ushort_t N>
    using  static_ushort=boost::mpl::integral_c<ushort_t,N>;
#else
    typedef int            int_t;
    typedef char           short_t;
    typedef unsigned int   uint_t;
    typedef unsigned short  ushort_t;
    template<int_t N>
    struct static_int : boost::mpl::integral_c<int_t,N>{};
    template<uint_t N>
    struct static_uint : boost::mpl::integral_c<uint_t,N>{};
    template<short_t N>
    struct static_short : boost::mpl::integral_c<short_t,N>{};
    template<ushort_t N>
    struct static_ushort : boost::mpl::integral_c<ushort_t,N>{};
#endif
