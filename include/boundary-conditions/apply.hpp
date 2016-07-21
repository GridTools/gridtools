/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
        for (uint_t i = halo_descriptors[0].loop_low_bound_outside(Direction::I);                               \
             i <= halo_descriptors[0].loop_high_bound_outside(Direction::I);                                    \
             ++i) {                                                                                             \
            for (uint_t j = halo_descriptors[1].loop_low_bound_outside(Direction::J);                           \
                 j <= halo_descriptors[1].loop_high_bound_outside(Direction::J);                                \
                 ++j) {                                                                                         \
                for (uint_t k = halo_descriptors[2].loop_low_bound_outside(Direction::K);                       \
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
