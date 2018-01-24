/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#define DEFS_GUARD

#if !defined(__CUDACC__)
#define CUDA8
#else
#if (GT_CUDA_VERSION > 75)
#define CUDA8
#endif
#endif

#if !defined(FUSION_MAX_VECTOR_SIZE)
#define FUSION_MAX_VECTOR_SIZE 20
#define FUSION_MAX_MAP_SIZE 20
#endif

#include <boost/mpl/for_each.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/vector.hpp>
#include <vector>

/**
   @file
   @brief global definitions
*/
#include <boost/mpl/bool.hpp>
#include <boost/mpl/logical.hpp>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#define GT_MAX_ARGS 20
#define GT_MAX_INDEPENDENT 3
#define GT_MAX_MSS 10

#if __cplusplus >= 201402L // since c++14
#define DEPRECATED(func) [[deprecated]] func
#define DEPRECATED_REASON(func, msg) [[deprecated(#msg)]] func
#else
#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__((deprecated))
#define DEPRECATED_REASON(func, msg) DEPRECATED(func)
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#define DEPRECATED_REASON(func, msg) DEPRECATED(func)
#else
#ifndef SUPPRESS_MESSAGES
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#endif
#define DEPRECATED(func) func
#define DEPRECATED_REASON(func, msg) DEPRECATED(func)
#endif
#endif

/**
 * Macro to allow make functions constexpr in c++14 (in case they are not only a return statement)
 */
#if __cplusplus >= 201402L
#define GT_CXX14CONSTEXPR constexpr
#else
#define GT_CXX14CONSTEXPR
#endif

/** Macro to enable additional checks that may catch some errors in user code
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

// max limit of indices for metastorages, beyond indices are reserved for library
#ifndef META_STORAGE_INDEX_LIMIT
#define META_STORAGE_INDEX_LIMIT 1000
#endif

#if defined(_OPENMP)
#include <omp.h>
#else
namespace gridtools {
    typedef int omp_int_t;
    inline omp_int_t omp_get_thread_num() { return 0; }
    inline omp_int_t omp_get_max_threads() { return 1; }
    inline double omp_get_wtime() { return 0; }
} // namespace gridtools
#endif

#include <boost/mpl/integral_c.hpp>
// macro defining empty copy constructors and assignment operators
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &);            \
    TypeName &operator=(const TypeName &)

namespace gridtools {
    /** \namespace enumtype
       @brief enumeration types*/
    namespace enumtype {
/**
   @section enumtypes Gridtools enumeration types
   @{
 */
/** enum specifying the type of backend we use */
#ifndef PLATFORM_GUARD
        enum platform { Cuda, Host };
#endif

        enum strategy { Naive, Block };

        /** enum specifying the type of grid to use */
        enum grid_type { structured, icosahedral };

        /** struct in order to perform templated methods partial specialization (Alexantrescu's trick, pre-c++11)*/
        template < typename EnumType, EnumType T >
        struct enum_type {
            static const EnumType value = T;
        };

        template < typename Value >
        struct is_enum {
            template < typename T >
            struct of_type {
                typedef typename boost::is_same< Value, enum_type< T, Value::value > >::type type;
                BOOST_STATIC_CONSTANT(bool, value = (type::value));
            };
        };

        enum isparallel { parallel_impl, serial };
        enum execution { forward, backward, parallel };

        template < enumtype::isparallel T, enumtype::execution U = forward >
        struct execute_impl {
            static const enumtype::execution iteration = U;
            static const enumtype::isparallel execution = T;
        };

        template < enumtype::execution U >
        struct execute {
            typedef execute_impl< serial, U > type;
        };

        template <>
        struct execute< parallel > {
            typedef execute_impl< parallel_impl, forward > type;
        };

        /*
         * accessor I/O policy
         */
        enum intend { in, inout };

#ifdef __CUDACC__
        static const unsigned int vector_width = 32;
#else
        static const unsigned int vector_width = 4;
#endif
        static const unsigned int metastorage_library_indices_limit = META_STORAGE_INDEX_LIMIT;

    } // namespace enumtype

#ifdef STRUCTURED_GRIDS
#define GRIDBACKEND structured
#else
#define GRIDBACKEND icosahedral
#endif

    template < typename Arg >
    struct is_enum_type : public boost::mpl::and_< typename boost::mpl::not_< boost::is_arithmetic< Arg > >::type,
                              typename boost::is_convertible< Arg, const int >::type >::type {};

    template < typename Arg1, typename Arg2 >
    struct any_enum_type : public boost::mpl::or_< is_enum_type< Arg1 >, is_enum_type< Arg2 > >::type {};

    template < typename T >
    struct is_backend_enum : boost::mpl::false_ {};

    /** checking that no arithmetic operation is performed on enum types*/
    template <>
    struct is_backend_enum< enumtype::platform > : boost::mpl::true_ {};

    struct error_no_operator_overload {};

    template < typename ArgType1,
        typename ArgType2,
        typename boost::enable_if< typename any_enum_type< ArgType1, ArgType2 >::type, int >::type = 0 >
    error_no_operator_overload operator+(ArgType1 arg1, ArgType2 arg2) {
        return {};
    }

    template < typename ArgType1,
        typename ArgType2,
        typename boost::enable_if< typename any_enum_type< ArgType1, ArgType2 >::type, int >::type = 0 >
    error_no_operator_overload operator-(ArgType1 arg1, ArgType2 arg2) {
        return {};
    }

    template < typename ArgType1,
        typename ArgType2,
        typename boost::enable_if< typename any_enum_type< ArgType1, ArgType2 >::type, int >::type = 0 >
    error_no_operator_overload operator*(ArgType1 arg1, ArgType2 arg2) {
        return {};
    }

    template < typename ArgType1,
        typename ArgType2,
        typename boost::enable_if< typename any_enum_type< ArgType1, ArgType2 >::type, int >::type = 0 >
    error_no_operator_overload operator/(ArgType1 arg1, ArgType2 arg2) {
        return {};
    }

    template < typename T >
    struct is_execution_engine : boost::mpl::false_ {};

    template < enumtype::execution U >
    struct is_execution_engine< enumtype::execute< U > > : boost::mpl::true_ {};

#define GT_WHERE_AM_I std::cout << __PRETTY_FUNCTION__ << " " << __FILE__ << ":" << __LINE__ << std::endl;

#define GRIDTOOLS_STATIC_ASSERT(Condition, Message) static_assert((Condition), "\n\nGRIDTOOLS ERROR=> " Message "\n\n")

#define GT_INTERNAL_ERROR                                                                                       \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team"

#define GT_INTERNAL_ERROR_MSG(x)                                                                                \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team. \nMessage\n\n" x

#define GT_AUTO_RETURN(expr)          \
    ->decltype(expr) { return expr; } \
    static_assert(1, "")

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
#define ASSERT_REAL_EQ(reference, actual) ASSERT_FLOAT_EQ(reference, actual)
#define EXPECT_REAL_EQ(reference, actual) EXPECT_FLOAT_EQ(reference, actual)
#elif FLOAT_PRECISION == 8
    typedef double float_type;
#define ASSERT_REAL_EQ(reference, actual) ASSERT_DOUBLE_EQ(reference, actual)
#define EXPECT_REAL_EQ(reference, actual) EXPECT_DOUBLE_EQ(reference, actual)
#else
#error float precision not properly set (4 or 8 bytes supported)
#endif

    // define a gridtools notype for metafunctions that would return something like void
    // but still to point to a real integral type so that it can be passed as argument to functions
    typedef int notype;

    using int_t = int;
    using short_t = int;
    using uint_t = unsigned int;
    using ushort_t = unsigned int;
    template < int_t N >
    using static_int = boost::mpl::integral_c< int_t, N >;
    template < uint_t N >
    using static_uint = boost::mpl::integral_c< uint_t, N >;
    template < short_t N >
    using static_short = boost::mpl::integral_c< short_t, N >;
    template < ushort_t N >
    using static_ushort = boost::mpl::integral_c< ushort_t, N >;
    template < size_t N >
    using static_size_t = boost::mpl::integral_c< size_t, N >;
    template < bool B >
    using static_bool = boost::mpl::integral_c< bool, B >;

} // namespace gridtools
