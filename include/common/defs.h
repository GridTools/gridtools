#pragma once
/**
   @file
   @brief global definitions
*/

#ifdef FUSION_MAX_VECTOR_SIZE
#undef FUSION_MAX_VECTOR_SIZE
#endif

#define FUSION_MAX_VECTOR_SIZE 20

#define GT_MAX_ARGS 20
#define GT_MAX_INDEPENDENT 3
#define GT_MAX_MSS 10

#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#ifndef SUPPRESS_MESSAGES
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#endif
#define DEPRECATED(func) func
#endif

/** Macro do enable additional checks that may catch some errors in user code
 */
#define PEDANTIC

#define RESTRICT __restrict__

#define GT_NO_ERRORS 0
#define GT_ERROR_NO_TEMPS 1

#if __cplusplus > 199711L
#ifndef CXX11_DISABLE
#define CXX11_ENABLED
#endif
#define CXX11_DISABLED
#endif

#ifndef GT_DEFAULT_TILE
#ifndef SUPPRESS_MESSAGES
#pragma message("INFO: Using default tile size = 8")
#endif
#define GT_DEFAULT_TILE 8
#endif

#include <boost/mpl/integral_c.hpp>

namespace gridtools{  namespace enumtype{
        /**
           @section enumtypes Gridtools enumeration types
           @{
         */
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


        enum isparallel {parallel_impl, serial} ;
        enum execution  {forward, backward, parallel} ;

        template<enumtype::isparallel T, enumtype::execution U=forward>
        struct execute_impl{
            static const enumtype::execution iteration=U;
            static const enumtype::isparallel execution=T;
        };

        template<enumtype::execution U>
        struct execute
        {
            typedef execute_impl<serial, U> type;
        };


        template<>
        struct execute<parallel>
        {
            typedef execute_impl<parallel_impl, forward> type;
        };
        /**
           @}
         */
    }//namespace enumtype

#ifndef CXX11_ENABLED
#define constexpr
#endif

#define GT_WHERE_AM_I                           \
    std::cout << __PRETTY_FUNCTION__ << " "     \
    << __FILE__ << ":"                          \
    << __LINE__                                 \
    << std::endl;



#ifdef CXX11_ENABLED
#define GRIDTOOLS_STATIC_ASSERT(Condition, Message)    static_assert(Condition, "\n\nGRIDTOOLS ERROR=> " Message"\n\n");
#else
#define GRIDTOOLS_STATIC_ASSERT(Condition, Message)    BOOST_STATIC_ASSERT(Condition);
#endif



//################ Type aliases for GridTools ################

    /**
       @section typedefs Gridtools types definitions
       @{
       @NOTE: the integer types are all signed,
       also the ones which should be logically unsigned (uint_t). This is due
       to a GCC (4.8.2) bug which is preventing vectorization of nested loops
       with an unsigned iteration index.
       https://gcc.gnu.org/bugzilla/show_bug.cgi?id=48052
    */

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

#ifdef CXX11_ENABLED
    using int_t          = int;
    using short_t        = int;
    using uint_t         = long unsigned int;
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
    typedef long int                     int_t;
    typedef int                     short_t;
    typedef unsigned int                     uint_t;
    typedef unsigned int                     ushort_t;
    template<int_t N>
    struct static_int : boost::mpl::integral_c<int_t,N>{
        typedef boost::mpl::integral_c<int_t,N> type;
    };
    template<uint_t N>
    struct static_uint : boost::mpl::integral_c<uint_t,N>{
        typedef boost::mpl::integral_c<uint_t,N> type;
    };
    template<short_t N>
    struct static_short : boost::mpl::integral_c<short_t,N>{
        typedef boost::mpl::integral_c<short_t,N> type;
    };
    template<ushort_t N>
    struct static_ushort : boost::mpl::integral_c<ushort_t,N>{
        typedef boost::mpl::integral_c<ushort_t,N> type;
    };
    /**
       @}
     */
//######################################################
#endif
}//namespace gridtools
