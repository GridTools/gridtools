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

#include <cmath>

#include "../common/array.hpp"
#include "../common/cuda_util.hpp"
#include "../common/defs.hpp"
#include "../common/halo_descriptor.hpp"
#include "../common/integral_constant.hpp"
#include "direction.hpp"
#include "predicate.hpp"

namespace gridtools {

    /** \ingroup Boundary-Conditions
     * @{
     */

    namespace apply_gpu_impl_ {
        using threads_per_block_x_t = integral_constant<int_t, 8>;
        using threads_per_block_y_t = integral_constant<int_t, 32>;
        using threads_per_block_z_t = integral_constant<int_t, 1>;

        /// Since predicate is runtime evaluated possibly on host-only data, we need to evaluate it before passing it to
        /// the CUDA kernels
        struct precomputed_pred {
            template <typename Predicate>
            precomputed_pred(Predicate const &p) {
                init<minus_, minus_, minus_>(p);
                init<minus_, minus_, zero_>(p);
                init<minus_, minus_, plus_>(p);

                init<minus_, zero_, minus_>(p);
                init<minus_, zero_, zero_>(p);
                init<minus_, zero_, plus_>(p);

                init<minus_, plus_, minus_>(p);
                init<minus_, plus_, zero_>(p);
                init<minus_, plus_, plus_>(p);

                init<zero_, minus_, minus_>(p);
                init<zero_, minus_, zero_>(p);
                init<zero_, minus_, plus_>(p);

                init<zero_, zero_, minus_>(p);

                init<zero_, zero_, plus_>(p);

                init<zero_, plus_, minus_>(p);
                init<zero_, plus_, zero_>(p);
                init<zero_, plus_, plus_>(p);

                init<plus_, minus_, minus_>(p);
                init<plus_, minus_, zero_>(p);
                init<plus_, minus_, plus_>(p);

                init<plus_, zero_, minus_>(p);
                init<plus_, zero_, zero_>(p);
                init<plus_, zero_, plus_>(p);

                init<plus_, plus_, minus_>(p);
                init<plus_, plus_, zero_>(p);
                init<plus_, plus_, plus_>(p);
            }

            precomputed_pred(precomputed_pred const &) = default;

            template <gridtools::sign I, gridtools::sign J, gridtools::sign K>
            GT_FUNCTION bool operator()(direction<I, J, K>) const {
                static constexpr uint_t index = direction_index<I, J, K>();
                return (m_values >> index) & 0x1;
            }

          private:
            template <gridtools::sign I, gridtools::sign J, gridtools::sign K, class Predicate>
            void init(Predicate const &p) {
                static constexpr uint_t mask = 0x1 << direction_index<I, J, K>();
                m_values = (m_values & ~mask) | (p(direction<I, J, K>{}) ? mask : 0);
            }

            template <gridtools::sign I, gridtools::sign J, gridtools::sign K>
            GT_FUNCTION static constexpr uint_t direction_index() {
                return (static_cast<int>(I) + 1) * 9 + (static_cast<int>(J) + 1) * 3 + (static_cast<int>(K) + 1);
            }

            GT_STATIC_ASSERT(sizeof(uint_t) >= 4, GT_INTERNAL_ERROR);

            uint_t m_values;
        };

        /** This class contains the information needed to identify
            which elements to access when applying boundary conditions
            depending on what region is targeted. Regions are
            identified by gridtools::direction which are equivalent to
            unit vectors (in infinite-norm).

            A configuration is an array of 26 shapes (actually 27, to
            make the addressing easier). A shape identifies a region
            of the boundary, which also inform of starting point for
            an iteration relative to the thread-ids, sizes and
            permutation to not go out of bound in the kernel
            configuration.
         */
        struct kernel_configuration {
            struct shape_type {
                array<uint_t, 3> m_size;
                array<uint_t, 3> m_sorted;
                array<uint_t, 3> m_start;
                array<uint_t, 3> m_perm = {{0, 1, 2}};

                shape_type() = default;

                shape_type(uint_t x, uint_t y, uint_t z, uint_t s0, uint_t s1, uint_t s2)
                    : m_size{x, y, z}, m_sorted{m_size}, m_start{s0, s1, s2} {
                    array<uint_t, 3> forward_perm = {{0, 1, 2}};
                    // Performing a simple insertion sort to compute the sorted sizes
                    // and then recover the permutation needed in the cuda kernel
                    for (int i = 0; i < 3; ++i) {
                        for (int j = i; j < 3; ++j) {
                            if (m_sorted[i] <= m_sorted[j]) {
                                uint_t t = m_sorted[i];
                                m_sorted[i] = m_sorted[j];
                                m_sorted[j] = t;

                                t = forward_perm[i];
                                forward_perm[i] = forward_perm[j];
                                forward_perm[j] = t;
                            }
                        }
                    }
                    // This loops computes the permutation needed later.
                    // forward_perm tells in what poition the sorted size comes from,
                    // the final m_perm tells in which position a given size is going
                    // after the sorting. This is the information needed to map threads
                    // to dimensions, since threads will come from a sorted (by
                    // decreasing sizes) pool
                    for (int i = 0; i < 3; ++i) {
                        m_perm[forward_perm[i]] = i;
                    }
                }

                GT_FUNCTION
                uint_t max() const { return m_sorted[0]; }

                GT_FUNCTION
                uint_t min() const { return m_sorted[2]; }

                GT_FUNCTION
                uint_t median() const { return m_sorted[1]; }

                GT_FUNCTION
                uint_t size(uint_t i) const { return m_size[i]; }

                GT_FUNCTION
                uint_t perm(uint_t i) const { return m_perm[i]; }

                GT_FUNCTION
                uint_t start(uint_t i) const { return m_start[i]; }
            };

            array<array<array<shape_type, 3>, 3>, 3> sizes;

            /** Kernel configuration takes the halo descriptors and
                generates the shapes of all possible halos portions
                that may be exchanged in a halo exchange
                operatior. These pieces are encoded into shapes that
                are described above here.
             */
            kernel_configuration(array<halo_descriptor, 3> const &halos) {

                array<array<uint_t, 3>, 3> segments;
                array<array<uint_t, 3>, 3> starts;

                for (int i = 0; i < 3; ++i) {
                    segments[i][0] = halos[i].minus();
                    segments[i][1] = halos[i].end() - halos[i].begin() + 1;
                    segments[i][2] = halos[i].plus();

                    starts[i][0] = halos[i].begin() - halos[i].minus();
                    starts[i][1] = halos[i].begin();
                    starts[i][2] = halos[i].end() + 1;
                }

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            sizes[i][j][k] = shape_type(segments[0][i],
                                segments[1][j],
                                segments[2][k],
                                starts[0][i],
                                starts[1][j],
                                starts[2][k]);
                        }
                    }
                }
            }

            dim3 block_size() const {
                dim3 b(0, 0, 0);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            if (i != 1 or j != 1 or k != 1) {
                                b.x = std::max(b.x, (decltype(b.x))sizes[i][j][k].max());
                                b.y = std::max(b.y, (decltype(b.y))sizes[i][j][k].median());
                                b.z = std::max(b.z, (decltype(b.z))sizes[i][j][k].min());
                            }
                        }
                    }
                }
                assert((b.x > 0 || b.y > 0 || b.z > 0) && "all boundary extents are empty");
                return b;
            }

            dim3 blocks() const {
                dim3 b = block_size();
                dim3 t = threads_per_block();
                return {b.x == 0 ? 1 : (b.x + t.x - 1) / t.x,
                    b.y == 0 ? 1 : (b.y + t.y - 1) / t.y,
                    b.z == 0 ? 1 : (b.z + t.z - 1) / t.z};
            }

            dim3 threads_per_block() const {
                return {threads_per_block_x_t::value, threads_per_block_y_t::value, threads_per_block_z_t::value};
            };

            GT_FUNCTION
            shape_type const &shape(uint_t i, uint_t j, uint_t k) const { return sizes[i][j][k]; }
        };

/** The following macro substitute the code to apply the boundary
    function to the boundary portion identified by a particular
    direction. The thread ids should be permuted since the shape
    of the boundary portion may be transposed w.r.t. the kernel
    configuration.
 */
#define GT_RUN_BC_ON(x, y, z)                                                                                      \
    if (predicate(direction<x, y, z>())) {                                                                         \
        auto const &shape = conf.shape(static_cast<int>(x) + 1, static_cast<int>(y) + 1, static_cast<int>(z) + 1); \
        if ((i < shape.max()) && (j < shape.median()) && (k < shape.min())) {                                      \
            boundary_function(direction<x, y, z>{},                                                                \
                data_views...,                                                                                     \
                thread_along_axis(i, j, k, shape.perm(0)) + shape.start(0),                                        \
                thread_along_axis(i, j, k, shape.perm(1)) + shape.start(1),                                        \
                thread_along_axis(i, j, k, shape.perm(2)) + shape.start(2));                                       \
        }                                                                                                          \
    }                                                                                                              \
    static_assert(true, " ")

        GT_FUNCTION int thread_along_axis(int i, int j, int k, int axis) {
            assert(axis >= 0 && axis < 3);
            return axis == 0 ? i : axis == 1 ? j : k;
        }

        /**
           @brief kernel to appy boundary conditions to the data fields requested
         */
        template <typename BoundaryFunction, typename... DataViews>
        __global__ void loop_kernel(BoundaryFunction const boundary_function,
            apply_gpu_impl_::precomputed_pred const predicate,
            apply_gpu_impl_::kernel_configuration const conf,
            DataViews const... data_views) {
            const uint_t i = blockIdx.x * apply_gpu_impl_::threads_per_block_x_t::value + threadIdx.x;
            const uint_t j = blockIdx.y * apply_gpu_impl_::threads_per_block_y_t::value + threadIdx.y;
            const uint_t k = blockIdx.z * apply_gpu_impl_::threads_per_block_z_t::value + threadIdx.z;

            GT_RUN_BC_ON(minus_, minus_, minus_);
            GT_RUN_BC_ON(minus_, minus_, zero_);
            GT_RUN_BC_ON(minus_, minus_, plus_);

            GT_RUN_BC_ON(minus_, zero_, minus_);
            GT_RUN_BC_ON(minus_, zero_, zero_);
            GT_RUN_BC_ON(minus_, zero_, plus_);

            GT_RUN_BC_ON(minus_, plus_, minus_);
            GT_RUN_BC_ON(minus_, plus_, zero_);
            GT_RUN_BC_ON(minus_, plus_, plus_);

            GT_RUN_BC_ON(zero_, minus_, minus_);
            GT_RUN_BC_ON(zero_, minus_, zero_);
            GT_RUN_BC_ON(zero_, minus_, plus_);

            GT_RUN_BC_ON(zero_, zero_, minus_);

            GT_RUN_BC_ON(zero_, zero_, plus_);

            GT_RUN_BC_ON(zero_, plus_, minus_);
            GT_RUN_BC_ON(zero_, plus_, zero_);
            GT_RUN_BC_ON(zero_, plus_, plus_);

            GT_RUN_BC_ON(plus_, minus_, minus_);
            GT_RUN_BC_ON(plus_, minus_, zero_);
            GT_RUN_BC_ON(plus_, minus_, plus_);

            GT_RUN_BC_ON(plus_, zero_, minus_);
            GT_RUN_BC_ON(plus_, zero_, zero_);
            GT_RUN_BC_ON(plus_, zero_, plus_);

            GT_RUN_BC_ON(plus_, plus_, minus_);
            GT_RUN_BC_ON(plus_, plus_, zero_);
            GT_RUN_BC_ON(plus_, plus_, plus_);
        }

#undef GT_RUN_BC_ON

    } // namespace apply_gpu_impl_

    /**
       @brief definition of the functions which apply the boundary conditions (arbitrary functions having as argument
       the
       direction, an arbitrary number of data fields, and the coordinates ID) in the halo region, see \ref
       gridtools::halo_descriptor

       For GPUs the idea is to let a single kernel deal with all the 26 boundary areas. The kernel configuration
       will depend on the largest dimensions of these boundary areas. The configuration shape will have dimensions
       sorted by decreasing sizes.

       For this reason each boundary area dimensions will be sorted by decreasing sizes and then the permutation
       needed to map the threads to the coordinates to use in the user provided boundary operators are kept.

       The shape information is kept in \ref apply_gpu_impl_::kernel_configuration::shape class, while the kernel
       configuration and the collections of shapes to be accessed in the kernel are stored in the \ref
       apply_gpu_impl_::kernel_configuration class.

       The kernel will then apply the user provided boundary functions in order to all the areas one after the other.
    */
    template <typename BoundaryFunction,
        typename Predicate = default_predicate,
        typename HaloDescriptors = array<halo_descriptor, 3>>
    struct boundary_apply_gpu {
      private:
        HaloDescriptors m_halo_descriptors;
        apply_gpu_impl_::kernel_configuration m_conf;
        BoundaryFunction const m_boundary_function;
        Predicate m_predicate;

      public:
        boundary_apply_gpu(HaloDescriptors const &hd, Predicate predicate = Predicate())
            : m_halo_descriptors(hd), m_conf{m_halo_descriptors}, m_boundary_function(BoundaryFunction()),
              m_predicate(predicate) {}

        boundary_apply_gpu(HaloDescriptors const &hd, BoundaryFunction const &bf, Predicate predicate = Predicate())
            : m_halo_descriptors(hd), m_conf{m_halo_descriptors}, m_boundary_function(bf), m_predicate(predicate) {}

        /**
           @brief applies the boundary conditions looping on the halo region defined by the member parameter, in all
        possible directions.
        this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n
        (DataField0, Datafield1, DataField2, ...)
        */
        template <typename... DataFieldViews>
        void apply(DataFieldViews const &... data_field_views) const {
            apply_gpu_impl_::loop_kernel<<<m_conf.blocks(), m_conf.threads_per_block()>>>(
                m_boundary_function, apply_gpu_impl_::precomputed_pred{m_predicate}, m_conf, data_field_views...);
#ifndef NDEBUG
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#else
            GT_CUDA_CHECK(cudaGetLastError());
#endif
        }
    };

    /** @} */

} // namespace gridtools
