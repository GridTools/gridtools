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

/**
@file
@brief definition of the functions which apply the boundary conditions (arbitrary functions having as argument the
direation, an arbitrary number of data fields, and the coordinates ID) in the halo region, see \ref
gridtools::halo_descriptor
*/
namespace gridtools {

    namespace _impl {
        struct kernel_configuration {
            struct shape_type {
                array< uint_t, 3 > data;
                array< uint_t, 3 > m_perm = {{0, 1, 2}};

                GT_FUNCTION
                shape_type() = default;

                GT_FUNCTION
                shape_type(uint_t x, uint_t y, uint_t z) : data{x, y, z} {
                    array< uint_t, 3 > out{data};
                    for (int i = 0; i < 3; ++i) {
                        for (int j = i; j < 3; ++j) {
                            if (out[i] < out[j]) {
                                uint_t t = out[i];
                                out[i] = out[j];
                                out[j] = t;

                                t = m_perm[i];
                                m_perm[i] = m_perm[j];
                                m_perm[j] = t;
                            }
                        }
                    }
                }

                GT_FUNCTION
                uint_t x() const { return data[0]; }
                GT_FUNCTION
                uint_t y() const { return data[1]; }
                GT_FUNCTION
                uint_t z() const { return data[2]; }

                GT_FUNCTION
                uint_t max() const { return data[m_perm[0]]; }

                GT_FUNCTION
                uint_t min() const { return data[m_perm[2]]; }

                GT_FUNCTION
                uint_t median() const { return data[m_perm[1]]; }

                GT_FUNCTION
                uint_t perm(uint_t i) const { return m_perm[i]; }
            };

            array< uint_t, 3 > configuration;
            array< array< array< shape_type, 3 >, 3 >, 3 > sizes;

            GT_FUNCTION
            kernel_configuration() = default;

            GT_FUNCTION
            kernel_configuration(array< halo_descriptor, 3 > const &halos) : configuration{0, 0, 0} {

                array< array< uint_t, 3 >, 3 > segments;

                for (int i = 0; i < 3; ++i) {
                    segments[i][0] = halos[i].minus();
                    segments[i][1] = halos[i].end() - halos[i].begin() + 1;
                    segments[i][2] = halos[i].plus();
                }

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            sizes[i][j][k] = shape_type(segments[0][i], segments[1][j], segments[2][k]);
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

#define RUN_BC_ON(x, y, z)                                                                                       \
    if (predicate(direction< x, y, z >())) {                                                                     \
        auto const &shape = conf.shape(0, 0, 0);                                                                 \
        if ((th[0] < shape.max()) && (th[1] < shape.median()) && (th[2] < shape.min())) {                        \
            boundary_function(                                                                                   \
                direction< x, y, z >{}, data_views..., th[shape.perm(0)], th[shape.perm(1)], th[shape.perm(2)]); \
        }                                                                                                        \
    }                                                                                                            \
    static_assert(true, " ")

    /**
       @brief kernel to appy boundary conditions to the data fields requested
     */
    template < typename BoundaryFunction, typename Predicate, typename... DataViews >
    __global__ void loop_kernel(BoundaryFunction boundary_function,
        Predicate predicate,
        _impl::kernel_configuration conf,
        DataViews... data_views) {
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
        RUN_BC_ON(zero_, zero_, zero_);
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
        const dim3 threads;

      public:
        boundary_apply_gpu(HaloDescriptors const &hd, Predicate predicate = Predicate())
            : m_halo_descriptors(hd), m_conf{m_halo_descriptors}, m_boundary_function(BoundaryFunction()),
              m_predicate(predicate), threads(ntx, nty, ntz) {}

        boundary_apply_gpu(HaloDescriptors const &hd, BoundaryFunction const &bf, Predicate predicate = Predicate())
            : m_halo_descriptors(hd), m_boundary_function(bf), m_predicate(predicate), threads(ntx, nty, ntz) {}

        /**
           @brief applies the boundary conditions looping on the halo region defined by the member parameter, in all
        possible directions.
        this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n
        (DataField0, Datafield1, DataField2, ...)
        */
        template < typename... DataFieldViews >
        void apply(DataFieldViews &... data_field_views) const {
            uint_t nx = m_conf.configuration[0];
            uint_t ny = m_conf.configuration[1];
            uint_t nz = m_conf.configuration[2];
            uint_t nbx = (nx == 0) ? (1) : ((nx + ntx - 1) / ntx);
            uint_t nby = (ny == 0) ? (1) : ((ny + nty - 1) / nty);
            uint_t nbz = (nz == 0) ? (1) : ((nz + ntz - 1) / ntz);
            assert(nx > 0 || ny > 0 || nz > 0 && "all boundary extents are empty");
            dim3 blocks(nbx, nby, nbz);
            loop_kernel<<< blocks, threads >>>(m_boundary_function, m_predicate, m_conf, data_field_views...);
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
        // template < typename... DataFieldViews >
        // void apply(DataFieldViews const &... data_field_views) const {

        //     if (m_predicate(direction< minus_, minus_, minus_ >()))
        //         apply_it< direction< minus_, minus_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< minus_, minus_, zero_ >()))
        //         apply_it< direction< minus_, minus_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< minus_, minus_, plus_ >()))
        //         apply_it< direction< minus_, minus_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< minus_, zero_, minus_ >()))
        //         apply_it< direction< minus_, zero_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< minus_, zero_, zero_ >()))
        //         apply_it< direction< minus_, zero_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< minus_, zero_, plus_ >()))
        //         apply_it< direction< minus_, zero_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< minus_, plus_, minus_ >()))
        //         apply_it< direction< minus_, plus_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< minus_, plus_, zero_ >()))
        //         apply_it< direction< minus_, plus_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< minus_, plus_, plus_ >()))
        //         apply_it< direction< minus_, plus_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< zero_, minus_, minus_ >()))
        //         apply_it< direction< zero_, minus_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< zero_, minus_, zero_ >()))
        //         apply_it< direction< zero_, minus_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< zero_, minus_, plus_ >()))
        //         apply_it< direction< zero_, minus_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< zero_, zero_, minus_ >()))
        //         apply_it< direction< zero_, zero_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< zero_, zero_, plus_ >()))
        //         apply_it< direction< zero_, zero_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< zero_, plus_, minus_ >()))
        //         apply_it< direction< zero_, plus_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< zero_, plus_, zero_ >()))
        //         apply_it< direction< zero_, plus_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< zero_, plus_, plus_ >()))
        //         apply_it< direction< zero_, plus_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< plus_, minus_, minus_ >()))
        //         apply_it< direction< plus_, minus_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< plus_, minus_, zero_ >()))
        //         apply_it< direction< plus_, minus_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< plus_, minus_, plus_ >()))
        //         apply_it< direction< plus_, minus_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< plus_, zero_, minus_ >()))
        //         apply_it< direction< plus_, zero_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< plus_, zero_, zero_ >()))
        //         apply_it< direction< plus_, zero_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< plus_, zero_, plus_ >()))
        //         apply_it< direction< plus_, zero_, plus_ > >(data_field_views...);

        //     if (m_predicate(direction< plus_, plus_, minus_ >()))
        //         apply_it< direction< plus_, plus_, minus_ > >(data_field_views...);
        //     if (m_predicate(direction< plus_, plus_, zero_ >()))
        //         apply_it< direction< plus_, plus_, zero_ > >(data_field_views...);
        //     if (m_predicate(direction< plus_, plus_, plus_ >()))
        //         apply_it< direction< plus_, plus_, plus_ > >(data_field_views...);

        //     cudaDeviceSynchronize();
        // }
    };

} // namespace gridtools
