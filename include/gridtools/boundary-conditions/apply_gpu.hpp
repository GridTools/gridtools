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

#include "../common/defs.hpp"
#include "../common/array.hpp"
#include "../common/halo_descriptor.hpp"
#include "direction.hpp"
#include "predicate.hpp"

#ifdef GT_SINGLE_KERNEL_BC
namespace gridtools {

    /** \ingroup Boundary-Conditions
     * @{
     */

    namespace _impl {
        struct precomputed_pred {
            array< array< array< bool, 3 >, 3 >, 3 > m_values;

            template < typename Predicate >
            precomputed_pred(Predicate const &p) {
                m_values[0][0][0] = p(direction< minus_, minus_, minus_ >{});
                m_values[0][0][1] = p(direction< minus_, minus_, zero_ >{});
                m_values[0][0][2] = p(direction< minus_, minus_, plus_ >{});

                m_values[0][1][0] = p(direction< minus_, zero_, minus_ >{});
                m_values[0][1][1] = p(direction< minus_, zero_, zero_ >{});
                m_values[0][1][2] = p(direction< minus_, zero_, plus_ >{});

                m_values[0][2][0] = p(direction< minus_, plus_, minus_ >{});
                m_values[0][2][1] = p(direction< minus_, plus_, zero_ >{});
                m_values[0][2][2] = p(direction< minus_, plus_, plus_ >{});

                m_values[1][0][0] = p(direction< zero_, minus_, minus_ >{});
                m_values[1][0][1] = p(direction< zero_, minus_, zero_ >{});
                m_values[1][0][2] = p(direction< zero_, minus_, plus_ >{});

                m_values[1][1][0] = p(direction< zero_, zero_, minus_ >{});

                m_values[1][1][2] = p(direction< zero_, zero_, plus_ >{});

                m_values[1][2][0] = p(direction< zero_, plus_, minus_ >{});
                m_values[1][2][1] = p(direction< zero_, plus_, zero_ >{});
                m_values[1][2][2] = p(direction< zero_, plus_, plus_ >{});

                m_values[2][0][0] = p(direction< plus_, minus_, minus_ >{});
                m_values[2][0][1] = p(direction< plus_, minus_, zero_ >{});
                m_values[2][0][2] = p(direction< plus_, minus_, plus_ >{});

                m_values[2][1][0] = p(direction< plus_, zero_, minus_ >{});
                m_values[2][1][1] = p(direction< plus_, zero_, zero_ >{});
                m_values[2][1][2] = p(direction< plus_, zero_, plus_ >{});

                m_values[2][2][0] = p(direction< plus_, plus_, minus_ >{});
                m_values[2][2][1] = p(direction< plus_, plus_, zero_ >{});
                m_values[2][2][2] = p(direction< plus_, plus_, plus_ >{});
            }

            GT_FUNCTION
            precomputed_pred(precomputed_pred const &) = default;

            template < sign I, sign J, sign K >
            GT_FUNCTION bool operator()(direction< I, J, K >) const {
                return m_values[static_cast< int >(I) + 1][static_cast< int >(J) + 1][static_cast< int >(K) + 1];
            }
        };

        struct kernel_configuration {
            struct shape_type {
                array< uint_t, 3 > m_size;
                array< uint_t, 3 > m_sorted;
                array< uint_t, 3 > m_start;
                array< uint_t, 3 > m_perm = {{0, 1, 2}};

                GT_FUNCTION
                shape_type() = default;

                GT_FUNCTION
                shape_type(uint_t x, uint_t y, uint_t z, uint_t s0, uint_t s1, uint_t s2)
                    : m_size{x, y, z}, m_sorted{m_size}, m_start{s0, s1, s2} {
                    array< uint_t, 3 > forward_perm = {{0, 1, 2}};
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

            array< uint_t, 3 > configuration = {{0, 0, 0}};
            array< array< array< shape_type, 3 >, 3 >, 3 > sizes;

            GT_FUNCTION
            kernel_configuration(array< halo_descriptor, 3 > const &halos) {

                array< array< uint_t, 3 >, 3 > segments;
                array< array< uint_t, 3 >, 3 > starts;

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

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            if (i != 1 or j != 1 or k != 1) {
                                configuration[0] = max(configuration[0], sizes[i][j][k].max());
                                configuration[1] = max(configuration[1], sizes[i][j][k].median());
                                configuration[2] = max(configuration[2], sizes[i][j][k].min());
                            }
                        }
                    }
                }
            }

            GT_FUNCTION
            shape_type const &shape(uint_t i, uint_t j, uint_t k) const { return sizes[i][j][k]; }
        };
    } // namespace _impl

#define RUN_BC_ON(x, y, z)                                                                               \
    if (predicate(direction< x, y, z >())) {                                                             \
        auto const &shape =                                                                              \
            conf.shape(static_cast< int >(x) + 1, static_cast< int >(y) + 1, static_cast< int >(z) + 1); \
        if ((th[0] < shape.max()) && (th[1] < shape.median()) && (th[2] < shape.min())) {                \
            boundary_function(direction< x, y, z >{},                                                    \
                data_views...,                                                                           \
                th[shape.perm(0)] + shape.start(0),                                                      \
                th[shape.perm(1)] + shape.start(1),                                                      \
                th[shape.perm(2)] + shape.start(2));                                                     \
        }                                                                                                \
    }                                                                                                    \
    static_assert(true, " ")

    /**
       @brief kernel to appy boundary conditions to the data fields requested
     */
    template < typename BoundaryFunction, typename Halos, typename... DataViews >
    __global__ void loop_kernel(BoundaryFunction const boundary_function,
        _impl::precomputed_pred const predicate,
        _impl::kernel_configuration const conf,
        Halos const halos,
        DataViews const... data_views) {
        array< uint_t, 3 > th{blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y,
            blockIdx.z * blockDim.z + threadIdx.z};

        RUN_BC_ON(minus_, minus_, minus_);
        RUN_BC_ON(minus_, minus_, zero_);
        RUN_BC_ON(minus_, minus_, plus_);

        RUN_BC_ON(minus_, zero_, minus_);
        RUN_BC_ON(minus_, zero_, zero_);
        RUN_BC_ON(minus_, zero_, plus_);

        RUN_BC_ON(minus_, plus_, minus_);
        RUN_BC_ON(minus_, plus_, zero_);
        RUN_BC_ON(minus_, plus_, plus_);

        RUN_BC_ON(zero_, minus_, minus_);
        RUN_BC_ON(zero_, minus_, zero_);
        RUN_BC_ON(zero_, minus_, plus_);

        RUN_BC_ON(zero_, zero_, minus_);

        RUN_BC_ON(zero_, zero_, plus_);

        RUN_BC_ON(zero_, plus_, minus_);
        RUN_BC_ON(zero_, plus_, zero_);
        RUN_BC_ON(zero_, plus_, plus_);

        RUN_BC_ON(plus_, minus_, minus_);
        RUN_BC_ON(plus_, minus_, zero_);
        RUN_BC_ON(plus_, minus_, plus_);

        RUN_BC_ON(plus_, zero_, minus_);
        RUN_BC_ON(plus_, zero_, zero_);
        RUN_BC_ON(plus_, zero_, plus_);

        RUN_BC_ON(plus_, plus_, minus_);
        RUN_BC_ON(plus_, plus_, zero_);
        RUN_BC_ON(plus_, plus_, plus_);
    }

    /**
       @brief definition of the functions which apply the boundary conditions (arbitrary functions having as argument the
       direction, an arbitrary number of data fields, and the coordinates ID) in the halo region, see \ref
       gridtools::halo_descriptor

       For GPUs the idea is to let a single kernel deal with all the 26 boundary areas. The kernel configuration
       will depend on the largest dimensions of these boundary areas. The configuration shape will have dimensions
       sorted by decreasing sizes.

       For this reason each boundary area dimensions will be sorted by decreasing sizes and then the permutation
       needed to map the threads to the coordinates to use in the user provided boundary operators are kept.

       The shape information is kept in \ref _impl::kernel_configuration::shape class, while the kernel
       configuration and the collections of shapes to be accessed in the kernel are stored in the \ref
       _impl::kernel_configuration class.

       The kernel will then apply the user provided boundary functions in order to all the areas one after the other.
    */
    template < typename BoundaryFunction,
        typename Predicate = default_predicate,
        typename HaloDescriptors = array< halo_descriptor, 3 > >
    struct boundary_apply_gpu {
      private:
        HaloDescriptors m_halo_descriptors;
        _impl::kernel_configuration m_conf;
        BoundaryFunction const m_boundary_function;
        Predicate m_predicate;
        static const uint_t ntx = 8, nty = 32, ntz = 1;
        const dim3 threads{ntx, nty, ntz};

      public:
        boundary_apply_gpu(HaloDescriptors const &hd, Predicate predicate = Predicate())
            : m_halo_descriptors(hd), m_conf{m_halo_descriptors}, m_boundary_function(BoundaryFunction()),
              m_predicate(predicate) {}

        boundary_apply_gpu(HaloDescriptors const &hd, BoundaryFunction const &bf, Predicate predicate = Predicate())
            : m_halo_descriptors(hd), m_conf{m_halo_descriptors}, m_boundary_function(bf), m_predicate(predicate),
              threads(ntx, nty, ntz) {}

        /**
           @brief applies the boundary conditions looping on the halo region defined by the member parameter, in all
        possible directions.
        this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n
        (DataField0, Datafield1, DataField2, ...)
        */
        template < typename... DataFieldViews >
        void apply(DataFieldViews const &... data_field_views) const {
            uint_t nx = m_conf.configuration[0];
            uint_t ny = m_conf.configuration[1];
            uint_t nz = m_conf.configuration[2];
            uint_t nbx = (nx == 0) ? (1) : ((nx + ntx - 1) / ntx);
            uint_t nby = (ny == 0) ? (1) : ((ny + nty - 1) / nty);
            uint_t nbz = (nz == 0) ? (1) : ((nz + ntz - 1) / ntz);
            assert(nx > 0 || ny > 0 || nz > 0 && "all boundary extents are empty");
            dim3 blocks(nbx, nby, nbz);
            loop_kernel<<< blocks, threads >>>(m_boundary_function,
                _impl::precomputed_pred{m_predicate},
                m_conf,
                m_halo_descriptors,
                data_field_views...);
            cudaDeviceSynchronize();
#ifndef NDEBUG
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                fprintf(stderr, "CUDA ERROR: %s in %s at line %dn", cudaGetErrorString(error), __FILE__, __LINE__);
                exit(-1);
            }
#endif
        }
    };

    /** @} */

} // namespace gridtools
#else
/**
@file
@brief definition of the functions which apply the boundary conditions (arbitrary functions having as argument the
direction, an arbitrary number of data fields, and the coordinates ID) in the halo region, see \ref
gridtools::halo_descriptor
*/
namespace gridtools {

    /** \ingroup Boundary-Conditions
     * @{
        */

    /**
       @brief kernel to apply boundary conditions to the data fields requested
    */
        template < typename BoundaryFunction, typename Direction, typename... DataViews >
            __global__ void loop_kernel(BoundaryFunction boundary_function,
                                        Direction direction,
                                        uint_t starti,
                                        uint_t startj,
                                        uint_t startk,
                                        uint_t nx,
                                        uint_t ny,
                                        uint_t nz,
                                        DataViews... data_views) {
            uint_t i = blockIdx.x * blockDim.x + threadIdx.x;
            uint_t j = blockIdx.y * blockDim.y + threadIdx.y;
            uint_t k = blockIdx.z * blockDim.z + threadIdx.z;
            if ((i < nx) && (j < ny) && (k < nz)) {
                boundary_function(direction, data_views..., i + starti, j + startj, k + startk);
            }
        }

        /**
       Struct to apply user specified boundary condition cases on data fields.
       \tparam BoundaryFunction The user class defining the operations on the boundary. It must be copy constructible.
       \tparam HaloDescriptors  The type behaving as a read only array of halo descriptors
        */
        template < typename BoundaryFunction,
                   typename Predicate = default_predicate,
                   typename HaloDescriptors = array< halo_descriptor, 3 > >
            struct boundary_apply_gpu {
        private:
            HaloDescriptors halo_descriptors;
            BoundaryFunction const boundary_function;
            Predicate predicate;
            static const uint_t ntx = 8, nty = 32, ntz = 1;
            const dim3 threads;

        public:
            boundary_apply_gpu(HaloDescriptors const &hd, Predicate predicate = Predicate())
                : halo_descriptors(hd), boundary_function(BoundaryFunction()), predicate(predicate),
                  threads(ntx, nty, ntz) {}

            boundary_apply_gpu(HaloDescriptors const &hd, BoundaryFunction const &bf, Predicate predicate = Predicate())
                : halo_descriptors(hd), boundary_function(bf), predicate(predicate), threads(ntx, nty, ntz) {}

            /**
           @brief applies the boundary conditions looping on the halo region defined by the member parameter, in all
        possible directions.
        this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n
        (DataField0, Datafield1, DataField2, ...)
            */
            template < typename Direction, typename... DataFieldViews >
            void apply_it(DataFieldViews &... data_field_views) const {
                uint_t nx = halo_descriptors[0].loop_high_bound_outside(Direction::I) -
                    halo_descriptors[0].loop_low_bound_outside(Direction::I) + 1;
                uint_t ny = halo_descriptors[1].loop_high_bound_outside(Direction::J) -
                    halo_descriptors[1].loop_low_bound_outside(Direction::J) + 1;
                uint_t nz = halo_descriptors[2].loop_high_bound_outside(Direction::K) -
                    halo_descriptors[2].loop_low_bound_outside(Direction::K) + 1;
                uint_t nbx = (nx == 0) ? (1) : ((nx + ntx - 1) / ntx);
                uint_t nby = (ny == 0) ? (1) : ((ny + nty - 1) / nty);
                uint_t nbz = (nz == 0) ? (1) : ((nz + ntz - 1) / ntz);
                assert(nx > 0 || ny > 0 || nz > 0 && "all boundary extents are empty");
                dim3 blocks(nbx, nby, nbz);
                loop_kernel<<< blocks, threads >>>(boundary_function,
                                                   Direction(),
                                                   halo_descriptors[0].loop_low_bound_outside(Direction::I),
                                                   halo_descriptors[1].loop_low_bound_outside(Direction::J),
                                                   halo_descriptors[2].loop_low_bound_outside(Direction::K),
                                                   nx,
                                                   ny,
                                                   nz,
                                                   data_field_views...);
                cudaDeviceSynchronize();
#ifndef NDEBUG
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    fprintf(stderr, "CUDA ERROR: %s in %s at line %dn", cudaGetErrorString(error), __FILE__, __LINE__);
                    exit(-1);
                }
#endif
            }

            /**
           @brief this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0
           to n
           (DataField0, Datafield1, DataField2, ...)
            */
            template < typename... DataFieldViews >
            void apply(DataFieldViews const &... data_field_views) const {

                if (predicate(direction< minus_, minus_, minus_ >()))
                    apply_it< direction< minus_, minus_, minus_ > >(data_field_views...);
                if (predicate(direction< minus_, minus_, zero_ >()))
                    apply_it< direction< minus_, minus_, zero_ > >(data_field_views...);
                if (predicate(direction< minus_, minus_, plus_ >()))
                    apply_it< direction< minus_, minus_, plus_ > >(data_field_views...);

                if (predicate(direction< minus_, zero_, minus_ >()))
                    apply_it< direction< minus_, zero_, minus_ > >(data_field_views...);
                if (predicate(direction< minus_, zero_, zero_ >()))
                    apply_it< direction< minus_, zero_, zero_ > >(data_field_views...);
                if (predicate(direction< minus_, zero_, plus_ >()))
                    apply_it< direction< minus_, zero_, plus_ > >(data_field_views...);

                if (predicate(direction< minus_, plus_, minus_ >()))
                    apply_it< direction< minus_, plus_, minus_ > >(data_field_views...);
                if (predicate(direction< minus_, plus_, zero_ >()))
                    apply_it< direction< minus_, plus_, zero_ > >(data_field_views...);
                if (predicate(direction< minus_, plus_, plus_ >()))
                    apply_it< direction< minus_, plus_, plus_ > >(data_field_views...);

                if (predicate(direction< zero_, minus_, minus_ >()))
                    apply_it< direction< zero_, minus_, minus_ > >(data_field_views...);
                if (predicate(direction< zero_, minus_, zero_ >()))
                    apply_it< direction< zero_, minus_, zero_ > >(data_field_views...);
                if (predicate(direction< zero_, minus_, plus_ >()))
                    apply_it< direction< zero_, minus_, plus_ > >(data_field_views...);

                if (predicate(direction< zero_, zero_, minus_ >()))
                    apply_it< direction< zero_, zero_, minus_ > >(data_field_views...);
                if (predicate(direction< zero_, zero_, plus_ >()))
                    apply_it< direction< zero_, zero_, plus_ > >(data_field_views...);

                if (predicate(direction< zero_, plus_, minus_ >()))
                    apply_it< direction< zero_, plus_, minus_ > >(data_field_views...);
                if (predicate(direction< zero_, plus_, zero_ >()))
                    apply_it< direction< zero_, plus_, zero_ > >(data_field_views...);
                if (predicate(direction< zero_, plus_, plus_ >()))
                    apply_it< direction< zero_, plus_, plus_ > >(data_field_views...);

                if (predicate(direction< plus_, minus_, minus_ >()))
                    apply_it< direction< plus_, minus_, minus_ > >(data_field_views...);
                if (predicate(direction< plus_, minus_, zero_ >()))
                    apply_it< direction< plus_, minus_, zero_ > >(data_field_views...);
                if (predicate(direction< plus_, minus_, plus_ >()))
                    apply_it< direction< plus_, minus_, plus_ > >(data_field_views...);

                if (predicate(direction< plus_, zero_, minus_ >()))
                    apply_it< direction< plus_, zero_, minus_ > >(data_field_views...);
                if (predicate(direction< plus_, zero_, zero_ >()))
                    apply_it< direction< plus_, zero_, zero_ > >(data_field_views...);
                if (predicate(direction< plus_, zero_, plus_ >()))
                    apply_it< direction< plus_, zero_, plus_ > >(data_field_views...);

                if (predicate(direction< plus_, plus_, minus_ >()))
                    apply_it< direction< plus_, plus_, minus_ > >(data_field_views...);
                if (predicate(direction< plus_, plus_, zero_ >()))
                    apply_it< direction< plus_, plus_, zero_ > >(data_field_views...);
                if (predicate(direction< plus_, plus_, plus_ >()))
                    apply_it< direction< plus_, plus_, plus_ > >(data_field_views...);

                cudaDeviceSynchronize();
            }
        };

        /** @} */

    } // namespace gridtools
#endif
