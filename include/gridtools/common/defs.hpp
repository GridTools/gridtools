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

#ifdef __CUDACC__
#define GT_CONSTEXPR
#else
#define GT_CONSTEXPR constexpr
#endif

#define GT_RESTRICT __restrict__

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
