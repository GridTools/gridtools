#pragma once

#if __cplusplus > 199711L
#ifndef CXX11_DISABLE
#define CXX11_ENABLED
#else
#define CXX11_DISABLED
#endif
#endif

//defines how many threads participate to the (shared) memory initialization
//TODOCOSUNA This IS VERY VERY VERY DANGEROUS HERE
#define BLOCK_SIZE 32

// #include <boost/mpl/map/aux_/item.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/vector.hpp>

/**
   @file
   @brief global definitions
*/
#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/logical.hpp>
#include <boost/type_traits.hpp>

#ifndef FUSION_MAX_VECTOR_SIZE
    #define FUSION_MAX_VECTOR_SIZE 20
    #define FUSION_MAX_MAP_SIZE 20
#endif

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
#ifndef PEDANTIC_DISABLED
#define PEDANTIC
#endif

#define RESTRICT __restrict__

#define GT_NO_ERRORS 0
#define GT_ERROR_NO_TEMPS 1

#ifndef GT_DEFAULT_TILE_I
  #ifdef __CUDACC__
    #define GT_DEFAULT_TILE_I 32
  #else
    #define GT_DEFAULT_TILE_I 8
  #endif
#endif
#ifndef GT_DEFAULT_TILE_J
  #ifdef __CUDACC__
    #define GT_DEFAULT_TILE_J 8
  #else
    #define GT_DEFAULT_TILE_J 8
  #endif
#endif

#if defined(_OPENMP)
  #include <omp.h>
#else
  typedef int omp_int_t;
  inline omp_int_t omp_get_thread_num() { return 0;}
  inline omp_int_t omp_get_max_threads() { return 1;}
  inline double omp_get_wtime() { return 0;}
#endif

#include <boost/mpl/integral_c.hpp>
// macro defining empty copy constructors and assignment operators
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&);               \
    TypeName& operator=(const TypeName&)

namespace gridtools{
    /** \namespace enumtype
       @brief enumeration types*/
    namespace enumtype{
        /**
           @section enumtypes Gridtools enumeration types
           @{
         */
        /** enum specifying the type of backend we use */
        enum backend  {Cuda, Host};

        enum strategy  {Naive, Block};

        /** struct in order to perform templated methods partial specialization (Alexantrescu's trick, pre-c++11)*/
        template<typename EnumType, EnumType T>
        struct enum_type
        {
            static const EnumType value=T;
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


    template<typename Arg >
    struct is_enum_type : public boost::mpl::and_<
        typename boost::mpl::not_<boost::is_arithmetic<Arg> >::type
        , typename boost::is_convertible<Arg, const int>::type >::type {};

    template<typename Arg1, typename Arg2 >
    struct any_enum_type : public boost::mpl::or_<is_enum_type<Arg1>, is_enum_type<Arg2> >::type {};

    template<typename T>
    struct is_backend_enum : boost::mpl::false_ {};

#ifdef CXX11_ENABLED
    /** checking that no arithmetic operation is performed on enum types*/
    template<>
    struct is_backend_enum<enumtype::backend> : boost::mpl::true_ {};
    struct error_no_operator_overload{};

    template <typename  ArgType1, typename ArgType2,
              typename boost::enable_if<typename any_enum_type<ArgType1, ArgType2>::type, int  >::type = 0>
    error_no_operator_overload operator + (ArgType1 arg1, ArgType2 arg2){}

    template <typename  ArgType1, typename ArgType2,
              typename boost::enable_if<typename any_enum_type<ArgType1, ArgType2>::type, int  >::type = 0>
    error_no_operator_overload operator - (ArgType1 arg1, ArgType2 arg2){}

    template <typename  ArgType1, typename ArgType2,
              typename boost::enable_if<typename any_enum_type<ArgType1, ArgType2>::type, int  >::type = 0>
    error_no_operator_overload operator * (ArgType1 arg1, ArgType2 arg2){}

    template <typename  ArgType1, typename ArgType2,
              typename boost::enable_if<typename any_enum_type<ArgType1, ArgType2>::type, int  >::type = 0>
    error_no_operator_overload operator / (ArgType1 arg1, ArgType2 arg2){}
#endif

    template<typename T>
    struct is_execution_engine : boost::mpl::false_{};

    template<enumtype::execution U>
    struct is_execution_engine<enumtype::execute<U> > : boost::mpl::true_{};


#ifndef CXX11_ENABLED
#define constexpr
#endif

#define GT_WHERE_AM_I                           \
    std::cout << __PRETTY_FUNCTION__ << " "     \
    << __FILE__ << ":"                          \
    << __LINE__                                 \
    << std::endl;



#ifdef CXX11_ENABLED
#define GRIDTOOLS_STATIC_ASSERT(Condition, Message)    static_assert(Condition, "\n\nGRIDTOOLS ERROR=> " Message"\n\n")
#else
#define GRIDTOOLS_STATIC_ASSERT(Condition, Message)    BOOST_STATIC_ASSERT(Condition)
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
    template<bool B>
    using  static_bool=boost::mpl::integral_c<bool,B>;
#else
    typedef int                     int_t;
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
    template<bool B>
    struct static_bool : boost::mpl::integral_c<bool,B>{
        typedef boost::mpl::integral_c<bool,B> type;
    };

    /**
       @}
     */
//######################################################
#endif

}//namespace gridtools
