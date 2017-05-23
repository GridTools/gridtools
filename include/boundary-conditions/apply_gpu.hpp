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
#include "direction.hpp"
#include "predicate.hpp"
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>

#include <boost/preprocessor/facilities/intercept.hpp>

/**
@file
@brief definition of the functions which apply the boundary conditions (arbitrary functions having as argument the
direation, an arbitrary number of data fields, and the coordinates ID) in the halo region, see \ref
gridtools::halo_descriptor
*/
namespace gridtools {

/**
   @brief this macro expands to n definitions of the function loop_kernel, taking a number of arguments ranging from 0
   to n (DataField0, Datafield1, DataField2, ...)
*/
#define GTLOOP(zz, n, nil)                                                                                         \
    template < typename BoundaryFunction,                                                                          \
        typename Direction,                                                                                        \
        BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField) >                                                \
    __global__ void loop_kernel(BoundaryFunction boundary_function,                                                \
        Direction direction,                                                                                       \
        BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, data_field),                                       \
        uint_t starti,                                                                                             \
        uint_t startj,                                                                                             \
        uint_t startk,                                                                                             \
        uint_t nx,                                                                                                 \
        uint_t ny,                                                                                                 \
        uint_t nz) {                                                                                               \
        uint_t i = blockIdx.x * blockDim.x + threadIdx.x;                                                          \
        uint_t j = blockIdx.y * blockDim.y + threadIdx.y;                                                          \
        uint_t k = blockIdx.z * blockDim.z + threadIdx.z;                                                          \
        if ((i < nx) && (j < ny) && (k < nz)) {                                                                    \
            boundary_function(                                                                                     \
                direction, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field), i + starti, j + startj, k + startk); \
        }                                                                                                          \
    }

    BOOST_PP_REPEAT(GT_MAX_ARGS, GTLOOP, _)

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
   @brief this macro expands to n definition of the function apply_it, taking a number of arguments ranging from 0 to n
   (DataField0, Datafield1, DataField2, ...)
*/
#define GTAPPLY_IT(z, n, nil)                                                                   \
    template < typename Direction, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField) >  \
    void apply_it(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, &data_field)) const { \
        uint_t nx = halo_descriptors[0].loop_high_bound_outside(Direction::I) -                 \
                    halo_descriptors[0].loop_low_bound_outside(Direction::I) + 1;               \
        uint_t ny = halo_descriptors[1].loop_high_bound_outside(Direction::J) -                 \
                    halo_descriptors[1].loop_low_bound_outside(Direction::J) + 1;               \
        uint_t nz = halo_descriptors[2].loop_high_bound_outside(Direction::K) -                 \
                    halo_descriptors[2].loop_low_bound_outside(Direction::K) + 1;               \
        uint_t nbx = (nx == 0) ? (1) : ((nx + ntx - 1) / ntx);                                  \
        uint_t nby = (ny == 0) ? (1) : ((ny + nty - 1) / nty);                                  \
        uint_t nbz = (nz == 0) ? (1) : ((nz + ntz - 1) / ntz);                                  \
        assert(nx > 0 || ny > 0 || nz > 0 && "all boundary extents are empty");                 \
        dim3 blocks(nbx, nby, nbz);                                                             \
        loop_kernel<<< blocks, threads >>>(boundary_function,                               \
            Direction(),                                                                        \
            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field),                                  \
            halo_descriptors[0].loop_low_bound_outside(Direction::I),                           \
            halo_descriptors[1].loop_low_bound_outside(Direction::J),                           \
            halo_descriptors[2].loop_low_bound_outside(Direction::K),                           \
            nx,                                                                                 \
            ny,                                                                                 \
            nz);                                                                                \
    }

#define GTAPPLY_IT_DEBUG(z, n, nil)                                                                              \
    template < typename Direction, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField) >                   \
    void apply_it(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, &data_field)) const {                  \
        uint_t nx = halo_descriptors[0].loop_high_bound_outside(Direction::I) -                                  \
                    halo_descriptors[0].loop_low_bound_outside(Direction::I) + 1;                                \
        uint_t ny = halo_descriptors[1].loop_high_bound_outside(Direction::J) -                                  \
                    halo_descriptors[1].loop_low_bound_outside(Direction::J) + 1;                                \
        uint_t nz = halo_descriptors[2].loop_high_bound_outside(Direction::K) -                                  \
                    halo_descriptors[2].loop_low_bound_outside(Direction::K) + 1;                                \
        uint_t nbx = (nx == 0) ? (1) : ((nx + ntx - 1) / ntx);                                                   \
        uint_t nby = (ny == 0) ? (1) : ((ny + nty - 1) / nty);                                                   \
        uint_t nbz = (nz == 0) ? (1) : ((nz + ntz - 1) / ntz);                                                   \
        assert(nx > 0 || ny > 0 || nz > 0 && "all boundary extents are empty");                                  \
        dim3 blocks(nbx, nby, nbz);                                                                              \
        loop_kernel<<< blocks, threads >>>(boundary_function,                                                \
            Direction(),                                                                                         \
            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field),                                                   \
            halo_descriptors[0].loop_low_bound_outside(Direction::I),                                            \
            halo_descriptors[1].loop_low_bound_outside(Direction::J),                                            \
            halo_descriptors[2].loop_low_bound_outside(Direction::K),                                            \
            nx,                                                                                                  \
            ny,                                                                                                  \
            nz);                                                                                                 \
        cudaDeviceSynchronize();                                                                                 \
        cudaError_t error = cudaGetLastError();                                                                  \
        if (error != cudaSuccess) {                                                                              \
            fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(-1);                                                                                            \
        }                                                                                                        \
    }

#ifdef NDEBUG
        BOOST_PP_REPEAT(GT_MAX_ARGS, GTAPPLY_IT, _)
#else
        BOOST_PP_REPEAT(GT_MAX_ARGS, GTAPPLY_IT_DEBUG, _)
#endif

/**
   @brief this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n
   (DataField0, Datafield1, DataField2, ...)
*/
#define GTAPPLY(z, n, nil)                                                                                      \
    template < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField) >                                      \
    void apply(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, const &data_field)) const {              \
                                                                                                                \
        if (predicate(direction< minus_, minus_, minus_ >()))                                                   \
            apply_it< direction< minus_, minus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
        if (predicate(direction< minus_, minus_, zero_ >()))                                                    \
            apply_it< direction< minus_, minus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< minus_, minus_, plus_ >()))                                                    \
            apply_it< direction< minus_, minus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
                                                                                                                \
        if (predicate(direction< minus_, zero_, minus_ >()))                                                    \
            apply_it< direction< minus_, zero_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< minus_, zero_, zero_ >()))                                                     \
            apply_it< direction< minus_, zero_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< minus_, zero_, plus_ >()))                                                     \
            apply_it< direction< minus_, zero_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                \
        if (predicate(direction< minus_, plus_, minus_ >()))                                                    \
            apply_it< direction< minus_, plus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< minus_, plus_, zero_ >()))                                                     \
            apply_it< direction< minus_, plus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< minus_, plus_, plus_ >()))                                                     \
            apply_it< direction< minus_, plus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                \
        if (predicate(direction< zero_, minus_, minus_ >()))                                                    \
            apply_it< direction< zero_, minus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< zero_, minus_, zero_ >()))                                                     \
            apply_it< direction< zero_, minus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< zero_, minus_, plus_ >()))                                                     \
            apply_it< direction< zero_, minus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                \
        if (predicate(direction< zero_, zero_, minus_ >()))                                                     \
            apply_it< direction< zero_, zero_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< zero_, zero_, plus_ >()))                                                      \
            apply_it< direction< zero_, zero_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
                                                                                                                \
        if (predicate(direction< zero_, plus_, minus_ >()))                                                     \
            apply_it< direction< zero_, plus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< zero_, plus_, zero_ >()))                                                      \
            apply_it< direction< zero_, plus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
        if (predicate(direction< zero_, plus_, plus_ >()))                                                      \
            apply_it< direction< zero_, plus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
                                                                                                                \
        if (predicate(direction< plus_, minus_, minus_ >()))                                                    \
            apply_it< direction< plus_, minus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< plus_, minus_, zero_ >()))                                                     \
            apply_it< direction< plus_, minus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< plus_, minus_, plus_ >()))                                                     \
            apply_it< direction< plus_, minus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                \
        if (predicate(direction< plus_, zero_, minus_ >()))                                                     \
            apply_it< direction< plus_, zero_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< plus_, zero_, zero_ >()))                                                      \
            apply_it< direction< plus_, zero_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
        if (predicate(direction< plus_, zero_, plus_ >()))                                                      \
            apply_it< direction< plus_, zero_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
                                                                                                                \
        if (predicate(direction< plus_, plus_, minus_ >()))                                                     \
            apply_it< direction< plus_, plus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< plus_, plus_, zero_ >()))                                                      \
            apply_it< direction< plus_, plus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
        if (predicate(direction< plus_, plus_, plus_ >()))                                                      \
            apply_it< direction< plus_, plus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
                                                                                                                \
        cudaDeviceSynchronize();                                                                                \
    }

        BOOST_PP_REPEAT(GT_MAX_ARGS, GTAPPLY, _)
    };

#undef GTLOOP
#undef GTAPPLY_IT
#undef GTAPPLY

} // namespace gridtools
