#pragma once
/**
   @file
   @brief global definitions
*/

#define GT_MAX_ARGS 10
#define GT_MAX_INDEPENDENT 3

#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif

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
    }//namespace enumtype
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

#include <boost/mpl/integral_c.hpp>
#ifdef CXX11_ENABLED
using int_t          = int;
using short_t        = int;
using uint_t         = unsigned int;
using ushort_t       = unsigned int;
template<int_t N>
using  static_int=boost::mpl::integral_c<int_t,N>;
template<uint_t N>
using  static_uint=boost::mpl::integral_c<uint_t,N>;
template<short_t N>
using  static_short=boost::mpl::integral_c<short_t,N>;
template<ushort_t N>
using  static_ushort=boost::mpl::integral_c<ushort_t,N>;
#else
typedef int                     int_t;
typedef int                          short_t;
typedef unsigned int            uint_t;
typedef unsigned int                 ushort_t;
template<int_t N>
struct static_int : boost::mpl::integral_c<int_t,N>{
    typedef boost::mpl::integral_c<uint_t,N> type;
};
template<uint_t N>
struct static_uint : boost::mpl::integral_c<uint_t,N>{
    typedef boost::mpl::integral_c<uint_t,N> type;
};
template<short_t N>
struct static_short : boost::mpl::integral_c<short_t,N>{
    typedef boost::mpl::integral_c<uint_t,N> type;
};
template<ushort_t N>
struct static_ushort : boost::mpl::integral_c<ushort_t,N>{
    typedef boost::mpl::integral_c<uint_t,N> type;
};

#endif
