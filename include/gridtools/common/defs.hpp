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

#include <type_traits>

#include "./generic_metafunctions/mpl_tags.hpp"

/** \ingroup common
    @{
    \defgroup defs Common Definitions
    @{
*/

/**
   @file
   @brief global definitions
*/

#define GT_MAX_ARGS 20
#define GT_MAX_INDEPENDENT 3
#define GT_MAX_MSS 10

#if __cplusplus >= 201402L // since c++14
#define GT_DEPRECATED(msg) [[deprecated(msg)]]
#else
#ifdef __GNUC__
#define GT_DEPRECATED(msg) __attribute__((deprecated))
#elif defined(_MSC_VER)
#define GT_DEPRECATED(msg) __declspec(deprecated)
#else
#ifndef SUPPRESS_MESSAGES
#pragma message("WARNING: You need to implement GT_DEPRECATED for this compiler")
#endif
#define GT_DEPRECATED(msg)
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

// macro defining empty copy constructors and assignment operators
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &);            \
    TypeName &operator=(const TypeName &)

// some compilers have the problem that template alias instantiations have exponential complexity
#if !defined(GT_BROKEN_TEMPLATE_ALIASES)
#if defined(__CUDACC_VER_MAJOR__)
// CUDA 9.0 and 9.1 have an different problem (not related to the exponential complexity of template alias
// instantiation) see https://github.com/eth-cscs/gridtools/issues/976
#define GT_BROKEN_TEMPLATE_ALIASES (__CUDACC_VER_MAJOR__ < 9)
#elif defined(__INTEL_COMPILER)
#define GT_BROKEN_TEMPLATE_ALIASES (__INTEL_COMPILER < 1800)
#elif defined(__clang__)
#define GT_BROKEN_TEMPLATE_ALIASES 0
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define GT_BROKEN_TEMPLATE_ALIASES (__GNUC__ * 10 + __GNUC_MINOR__ < 47)
#else
#define GT_BROKEN_TEMPLATE_ALIASES 1
#endif
#endif

// check boost::optional workaround for CUDA9.2
#if (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)
#if (not defined(BOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL) || \
     not defined(BOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE))
#error \
    "CUDA 9.2 has a problem with boost::optional, please define BOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL and BOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE prior to any include of boost/optional.hpp"
#endif
#endif

/**
 * @brief Main namespace containing all the provided libraries and
 * functionalities
 */
namespace gridtools {
    /** \ingroup defs
        @{
    */

    /** tags specifying the target to use */
    namespace target {
        struct cuda {};
        struct mc {};
        struct x86 {};
    } // namespace target

    /** tags specifying the strategy to use */
    namespace strategy {
        struct naive {};
        struct block {};
    } // namespace strategy

    /** tags specifying the type of grid to use */
    namespace grid_type {
        struct structured {};
        struct icosahedral {};
    } // namespace grid_type

    /** \namespace enumtype
       @brief enumeration types*/
    namespace enumtype {
        /**
           @section enumtypes Gridtools enumeration types
           @{
         */

        /*
         * accessor I/O policy
         */
        enum intent { in, inout };

#ifdef __CUDACC__
        static const unsigned int vector_width = 32;
#else
        static const unsigned int vector_width = 4;
#endif
        static const unsigned int metastorage_library_indices_limit = META_STORAGE_INDEX_LIMIT;

    } // namespace enumtype

#ifdef STRUCTURED_GRIDS
#define GRIDBACKEND gridtools::grid_type::structured
#else
#define GRIDBACKEND gridtools::grid_type::icosahedral
#endif

#define GRIDTOOLS_STATIC_ASSERT(Condition, Message) static_assert((Condition), "\n\nGRIDTOOLS ERROR=> " Message "\n\n")

#define GT_INTERNAL_ERROR                                                                                       \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team"

#define GT_INTERNAL_ERROR_MSG(x)                                                                                \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team. \nMessage\n\n" x

#ifdef GT_DOXYGEN_SHOULD_EXCLUDE_THIS
/* disable GT_AUTO_RETURN macro for doxygen as it creates many warnings */
#define GT_AUTO_RETURN(expr)
#else
#define GT_AUTO_RETURN(expr)          \
    ->decltype(expr) { return expr; } \
    static_assert(1, "")
#endif

    //################ Type aliases for GridTools ################

    /**
       @section typedefs Gridtools types definitions
       @{
       @note the integer types are all signed,
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
    template <int_t N>
    using static_int = std::integral_constant<int_t, N>;
    template <uint_t N>
    using static_uint = std::integral_constant<uint_t, N>;
    template <short_t N>
    using static_short = std::integral_constant<short_t, N>;
    template <ushort_t N>
    using static_ushort = std::integral_constant<ushort_t, N>;

    template <size_t N>
    using static_size_t = std::integral_constant<size_t, N>;
    template <bool B>
    using static_bool = std::integral_constant<bool, B>;

    /** @} */

} // namespace gridtools

/** @} */
/** @} */
