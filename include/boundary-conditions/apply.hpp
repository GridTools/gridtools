/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

/**
@file
@brief  definition of the functions which apply the boundary conditions (arbitrary functions having as argument the
direation, an arbitrary number of data fields, and the coordinates ID)
*/
namespace gridtools {

#ifndef CXX11_ENABLED

    /**
       Struct to apply user specified boundary condition cases on data fields.

       \tparam BoundaryFunction The user class defining the operations on the boundary. It must be copy constructible.
       \tparam HaloDescriptors  The type behaving as a read only array of halo descriptors
     */
    template < typename BoundaryFunction,
        typename Predicate = default_predicate,
        typename HaloDescriptors = array< halo_descriptor, 3 > >
    struct boundary_apply {
      private:
        HaloDescriptors halo_descriptors;
        BoundaryFunction const boundary_function;
        Predicate predicate;

/**
   @brief loops on the halo region defined by the HaloDescriptor member parameter, and evaluates the boundary_function
   in the specified direction, in the specified halo node.
   this macro expands to n definitions of the function loop, taking a number of arguments ranging from 0 to n
   (DataField0, Datafield1, DataField2, ...)
*/
#define GTLOOP(z, n, nil)                                                                                       \
    template < typename Direction, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField) >                  \
    void loop(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, &data_field)) const {                     \
        for (int_t i = halo_descriptors[0].loop_low_bound_outside(Direction::I);                                \
             i <= halo_descriptors[0].loop_high_bound_outside(Direction::I);                                    \
             ++i) {                                                                                             \
            for (int_t j = halo_descriptors[1].loop_low_bound_outside(Direction::J);                            \
                 j <= halo_descriptors[1].loop_high_bound_outside(Direction::J);                                \
                 ++j) {                                                                                         \
                for (int_t k = halo_descriptors[2].loop_low_bound_outside(Direction::K);                        \
                     k <= halo_descriptors[2].loop_high_bound_outside(Direction::K);                            \
                     ++k) {                                                                                     \
                    boundary_function(Direction(), BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field), i, j, k); \
                }                                                                                               \
            }                                                                                                   \
        }                                                                                                       \
    }

        BOOST_PP_REPEAT(GT_MAX_ARGS, GTLOOP, _)

      public:
        boundary_apply(HaloDescriptors const &hd, Predicate predicate = Predicate())
            : halo_descriptors(hd), boundary_function(BoundaryFunction()), predicate(predicate) {}

        boundary_apply(HaloDescriptors const &hd, BoundaryFunction const &bf, Predicate predicate = Predicate())
            : halo_descriptors(hd), boundary_function(bf), predicate(predicate) {}

/**
   @brief applies the boundary conditions looping on the halo region defined by the member parameter, in all possible
directions.
this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n (DataField0,
Datafield1, DataField2, ...)

*/
#define GTAPPLY(z, n, nil)                                                                                        \
    template < BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField) >                                        \
    void apply(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, &data_field)) const {                      \
                                                                                                                  \
        if (predicate(direction< minus_, minus_, minus_ >()))                                                     \
            this->loop< direction< minus_, minus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
        if (predicate(direction< minus_, minus_, zero_ >()))                                                      \
            this->loop< direction< minus_, minus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< minus_, minus_, plus_ >()))                                                      \
            this->loop< direction< minus_, minus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
                                                                                                                  \
        if (predicate(direction< minus_, zero_, minus_ >()))                                                      \
            this->loop< direction< minus_, zero_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< minus_, zero_, zero_ >()))                                                       \
            this->loop< direction< minus_, zero_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< minus_, zero_, plus_ >()))                                                       \
            this->loop< direction< minus_, zero_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                  \
        if (predicate(direction< minus_, plus_, minus_ >()))                                                      \
            this->loop< direction< minus_, plus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< minus_, plus_, zero_ >()))                                                       \
            this->loop< direction< minus_, plus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< minus_, plus_, plus_ >()))                                                       \
            this->loop< direction< minus_, plus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                  \
        if (predicate(direction< zero_, minus_, minus_ >()))                                                      \
            this->loop< direction< zero_, minus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< zero_, minus_, zero_ >()))                                                       \
            this->loop< direction< zero_, minus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< zero_, minus_, plus_ >()))                                                       \
            this->loop< direction< zero_, minus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                  \
        if (predicate(direction< zero_, zero_, minus_ >()))                                                       \
            this->loop< direction< zero_, zero_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< zero_, zero_, plus_ >()))                                                        \
            this->loop< direction< zero_, zero_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
                                                                                                                  \
        if (predicate(direction< zero_, plus_, minus_ >()))                                                       \
            this->loop< direction< zero_, plus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< zero_, plus_, zero_ >()))                                                        \
            this->loop< direction< zero_, plus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
        if (predicate(direction< zero_, plus_, plus_ >()))                                                        \
            this->loop< direction< zero_, plus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
                                                                                                                  \
        if (predicate(direction< plus_, minus_, minus_ >()))                                                      \
            this->loop< direction< plus_, minus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));  \
        if (predicate(direction< plus_, minus_, zero_ >()))                                                       \
            this->loop< direction< plus_, minus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< plus_, minus_, plus_ >()))                                                       \
            this->loop< direction< plus_, minus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
                                                                                                                  \
        if (predicate(direction< plus_, zero_, minus_ >()))                                                       \
            this->loop< direction< plus_, zero_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< plus_, zero_, zero_ >()))                                                        \
            this->loop< direction< plus_, zero_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
        if (predicate(direction< plus_, zero_, plus_ >()))                                                        \
            this->loop< direction< plus_, zero_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
                                                                                                                  \
        if (predicate(direction< plus_, plus_, minus_ >()))                                                       \
            this->loop< direction< plus_, plus_, minus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));   \
        if (predicate(direction< plus_, plus_, zero_ >()))                                                        \
            this->loop< direction< plus_, plus_, zero_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
        if (predicate(direction< plus_, plus_, plus_ >()))                                                        \
            this->loop< direction< plus_, plus_, plus_ > >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field));    \
    }

        BOOST_PP_REPEAT(GT_MAX_ARGS, GTAPPLY, _)
    };

#undef GTLOOP
#undef GTAPPLY

#else  //#ifndef CXX11_ENABLED

    template < typename BoundaryFunction,
        typename Predicate = default_predicate,
        typename HaloDescriptors = array< halo_descriptor, 3 > >
    struct boundary_apply {
      private:
        HaloDescriptors halo_descriptors;
        BoundaryFunction const boundary_function;
        Predicate predicate;

        /** @brief loops on the halo region defined by the HaloDescriptor member parameter, and evaluates the
           boundary_function in the specified direction, in the specified halo node.
            this macro expands to n definitions of the function loop, taking a number of arguments ranging from 0 to n
           (DataField0, Datafield1, DataField2, ...)*/
        template < typename Direction, typename... DataField >
        void loop(DataField &... data_field) const {
            for (int_t i = halo_descriptors[0].loop_low_bound_outside(Direction::I);
                 i <= halo_descriptors[0].loop_high_bound_outside(Direction::I);
                 ++i) {
                for (int_t j = halo_descriptors[1].loop_low_bound_outside(Direction::J);
                     j <= halo_descriptors[1].loop_high_bound_outside(Direction::J);
                     ++j) {
                    for (int_t k = halo_descriptors[2].loop_low_bound_outside(Direction::K);
                         k <= halo_descriptors[2].loop_high_bound_outside(Direction::K);
                         ++k) {
                        boundary_function(Direction(), data_field..., i, j, k);
                    }
                }
            }
        }

      public:
        boundary_apply(HaloDescriptors const &hd, Predicate predicate = Predicate())
            : halo_descriptors(hd), boundary_function(BoundaryFunction()), predicate(predicate) {}

        boundary_apply(HaloDescriptors const &hd, BoundaryFunction const &bf, Predicate predicate = Predicate())
            : halo_descriptors(hd), boundary_function(bf), predicate(predicate) {}

        /**
           @brief applies the boundary conditions looping on the halo region defined by the member parameter, in all
        possible directions.
        this macro expands to n definitions of the function apply, taking a number of arguments ranging from 0 to n
        (DataField0, Datafield1, DataField2, ...)

        */
        template < typename First, typename... DataFields >
        void apply(First &first, DataFields &... data_fields) const {

            if (predicate(direction< minus_, minus_, minus_ >()))
                this->loop< direction< minus_, minus_, minus_ > >(first, data_fields...);
            if (predicate(direction< minus_, minus_, zero_ >()))
                this->loop< direction< minus_, minus_, zero_ > >(first, data_fields...);
            if (predicate(direction< minus_, minus_, plus_ >()))
                this->loop< direction< minus_, minus_, plus_ > >(first, data_fields...);

            if (predicate(direction< minus_, zero_, minus_ >()))
                this->loop< direction< minus_, zero_, minus_ > >(first, data_fields...);
            if (predicate(direction< minus_, zero_, zero_ >()))
                this->loop< direction< minus_, zero_, zero_ > >(first, data_fields...);
            if (predicate(direction< minus_, zero_, plus_ >()))
                this->loop< direction< minus_, zero_, plus_ > >(first, data_fields...);

            if (predicate(direction< minus_, plus_, minus_ >()))
                this->loop< direction< minus_, plus_, minus_ > >(first, data_fields...);
            if (predicate(direction< minus_, plus_, zero_ >()))
                this->loop< direction< minus_, plus_, zero_ > >(first, data_fields...);
            if (predicate(direction< minus_, plus_, plus_ >()))
                this->loop< direction< minus_, plus_, plus_ > >(first, data_fields...);

            if (predicate(direction< zero_, minus_, minus_ >()))
                this->loop< direction< zero_, minus_, minus_ > >(first, data_fields...);
            if (predicate(direction< zero_, minus_, zero_ >()))
                this->loop< direction< zero_, minus_, zero_ > >(first, data_fields...);
            if (predicate(direction< zero_, minus_, plus_ >()))
                this->loop< direction< zero_, minus_, plus_ > >(first, data_fields...);

            if (predicate(direction< zero_, zero_, minus_ >()))
                this->loop< direction< zero_, zero_, minus_ > >(first, data_fields...);
            if (predicate(direction< zero_, zero_, plus_ >()))
                this->loop< direction< zero_, zero_, plus_ > >(first, data_fields...);

            if (predicate(direction< zero_, plus_, minus_ >()))
                this->loop< direction< zero_, plus_, minus_ > >(first, data_fields...);
            if (predicate(direction< zero_, plus_, zero_ >()))
                this->loop< direction< zero_, plus_, zero_ > >(first, data_fields...);
            if (predicate(direction< zero_, plus_, plus_ >()))
                this->loop< direction< zero_, plus_, plus_ > >(first, data_fields...);

            if (predicate(direction< plus_, minus_, minus_ >()))
                this->loop< direction< plus_, minus_, minus_ > >(first, data_fields...);
            if (predicate(direction< plus_, minus_, zero_ >()))
                this->loop< direction< plus_, minus_, zero_ > >(first, data_fields...);
            if (predicate(direction< plus_, minus_, plus_ >()))
                this->loop< direction< plus_, minus_, plus_ > >(first, data_fields...);

            if (predicate(direction< plus_, zero_, minus_ >()))
                this->loop< direction< plus_, zero_, minus_ > >(first, data_fields...);
            if (predicate(direction< plus_, zero_, zero_ >()))
                this->loop< direction< plus_, zero_, zero_ > >(first, data_fields...);
            if (predicate(direction< plus_, zero_, plus_ >()))
                this->loop< direction< plus_, zero_, plus_ > >(first, data_fields...);

            if (predicate(direction< plus_, plus_, minus_ >()))
                this->loop< direction< plus_, plus_, minus_ > >(first, data_fields...);
            if (predicate(direction< plus_, plus_, zero_ >()))
                this->loop< direction< plus_, plus_, zero_ > >(first, data_fields...);
            if (predicate(direction< plus_, plus_, plus_ >()))
                this->loop< direction< plus_, plus_, plus_ > >(first, data_fields...);

            // apply(data_fields ...);
        }

      private:
        /** fixing compilation */
        void apply() const {}
    };
#endif //#ifndef CXX11_ENABLED

} // namespace gridtools
