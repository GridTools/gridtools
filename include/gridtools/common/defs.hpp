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

#include "integral_constant.hpp"

/** \ingroup common
    @{
    \defgroup defs Common Definitions
    @{
*/

/**
   @file
   @brief global definitions
*/

//################ Type aliases for GridTools ################

#ifdef __NVCC__
#define GT_CONSTEXPR
#else
#define GT_CONSTEXPR constexpr
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

/**
 * @brief Main namespace containing all the provided libraries and
 * functionalities
 */
namespace gridtools {
    /** \ingroup defs
        @{
    */
    using int_t = int;
    using uint_t = unsigned int;

    template <int_t N>
    using static_int = std::integral_constant<int_t, N>;
    template <uint_t N>
    using static_uint = std::integral_constant<uint_t, N>;

    namespace naive {
        struct backend {};
    } // namespace naive

    namespace cuda {
        template <class IBlockSize = integral_constant<int_t, 64>,
            class JBlockSize = integral_constant<int_t, 8>,
            class KBlockSize = integral_constant<int_t, 1>>
        struct backend {
            using i_block_size_t = IBlockSize;
            using j_block_size_t = JBlockSize;
            using k_block_size_t = KBlockSize;

            static constexpr i_block_size_t i_block_size() { return {}; }
            static constexpr j_block_size_t j_block_size() { return {}; }
            static constexpr k_block_size_t k_block_size() { return {}; }
        };
    } // namespace cuda

    namespace mc {
        struct backend {};
    } // namespace mc

    namespace x86 {
        template <class IBlockSize = integral_constant<int_t, 8>, class JBlockSize = integral_constant<int_t, 8>>
        struct backend {
            using i_block_size_t = IBlockSize;
            using j_block_size_t = JBlockSize;

            static constexpr i_block_size_t i_block_size() { return {}; }
            static constexpr j_block_size_t j_block_size() { return {}; }
        };
    } // namespace x86

    /** tags specifying the backend to use */
    namespace backend {
        using cuda = cuda::backend<>;
        using mc = mc::backend;
        using x86 = x86::backend<>;
        using naive = naive::backend;
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

    /** @} */

} // namespace gridtools

/** @} */
/** @} */
