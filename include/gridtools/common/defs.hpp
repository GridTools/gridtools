/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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
#ifndef GT_PEDANTIC_DISABLED
#define GT_PEDANTIC
#endif

#define GT_RESTRICT __restrict__

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
#ifndef GT_META_STORAGE_INDEX_LIMIT
#define GT_META_STORAGE_INDEX_LIMIT 1000
#endif
static const unsigned int metastorage_library_indices_limit = GT_META_STORAGE_INDEX_LIMIT;

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

    /** tags specifying the backend to use */
    namespace backend {
        struct cuda {};
        struct mc {};
        struct x86 {};
        struct naive {};
    } // namespace backend

#define GT_STATIC_ASSERT(Condition, Message) static_assert((Condition), "\n\nGRIDTOOLS ERROR=> " Message "\n\n")

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

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ < 9 || __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ < 2)
#define GT_DECLARE_DEFAULT_EMPTY_CTOR(class_name)                          \
    __forceinline__ __host__ __device__ constexpr class_name() noexcept {} \
    static_assert(1, "")
#else
#define GT_DECLARE_DEFAULT_EMPTY_CTOR(class_name) class_name() = default
#endif

    //################ Type aliases for GridTools ################

    using int_t = int;
    using uint_t = unsigned int;
    template <int_t N>
    using static_int = std::integral_constant<int_t, N>;
    template <uint_t N>
    using static_uint = std::integral_constant<uint_t, N>;

    /** @} */

} // namespace gridtools

/** @} */
/** @} */
