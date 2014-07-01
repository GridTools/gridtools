#pragma once

#include "../common/defs.h"
#include "direction.h"
#include "predicate.h"

namespace gridtools {

    /**
       Struct to apply user specified boundary condition cases on data fields.

       \tparam BoundaryFunction The user class defining the operations on the boundary. It must be copy constructible.
       \tparam HaloDescriptors  The type behaving as a read only array of halo descriptors
     */
    template <typename BoundaryFunction, typename HaloDescriptors = array<halo_descriptor, 3>, typename Predicate = default_predicate >
    struct boundary_apply {
    private:
        HaloDescriptors halo_descriptors;
        BoundaryFunction const boundary_function;
        Predicate predicate;

#define GTLOOP(z, n, nil)                                               \
        template <typename Direction, BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField)> \
        void loop(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, & data_field)) const { \
            for (int i=halo_descriptors[0].loop_low_bound_outside(Direction::I); \
                 i<=halo_descriptors[0].loop_high_bound_outside(Direction::I); \
                 ++i) {                                                 \
                for (int j=halo_descriptors[1].loop_low_bound_outside(Direction::J); \
                     j<=halo_descriptors[1].loop_high_bound_outside(Direction::J); \
                     ++j) {                                             \
                    for (int k=halo_descriptors[2].loop_low_bound_outside(Direction::K); \
                         k<=halo_descriptors[2].loop_high_bound_outside(Direction::K); \
                         ++k) {                                         \
                        boundary_function(Direction(),                  \
                            BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field), i, j, k); \
                    }                                                   \
                }                                                       \
            }                                                           \
        }

        BOOST_PP_REPEAT(GT_MAX_ARGS, GTLOOP, _)

        public:
        boundary_apply(HaloDescriptors const& hd, Predicate predicate = Predicate() )
            : halo_descriptors(hd)
            , boundary_function(BoundaryFunction())
            , predicate(predicate)
        {}

        boundary_apply(HaloDescriptors const& hd, BoundaryFunction const & bf, Predicate predicate = Predicate() )
            : halo_descriptors(hd)
            , boundary_function(bf)
            , predicate(predicate)
        {}

#define GTAPPLY(z, n, nil)                                                \
        template <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField)> \
        void apply(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, & data_field) ) const { \
                                                                        \
            if (predicate(direction<minus,minus,minus>())) this->loop<direction<minus,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction<minus,minus, zero>())) this->loop<direction<minus,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction<minus,minus, plus>())) this->loop<direction<minus,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction<minus, zero,minus>())) this->loop<direction<minus, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction<minus, zero, zero>())) this->loop<direction<minus, zero, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction<minus, zero, plus>())) this->loop<direction<minus, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction<minus, plus,minus>())) this->loop<direction<minus, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction<minus, plus, zero>())) this->loop<direction<minus, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction<minus, plus, plus>())) this->loop<direction<minus, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction< zero,minus,minus>())) this->loop<direction< zero,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< zero,minus, zero>())) this->loop<direction< zero,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< zero,minus, plus>())) this->loop<direction< zero,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction< zero, zero,minus>())) this->loop<direction< zero, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< zero, zero, plus>())) this->loop<direction< zero, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction< zero, plus,minus>())) this->loop<direction< zero, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< zero, plus, zero>())) this->loop<direction< zero, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< zero, plus, plus>())) this->loop<direction< zero, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction< plus,minus,minus>())) this->loop<direction< plus,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< plus,minus, zero>())) this->loop<direction< plus,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< plus,minus, plus>())) this->loop<direction< plus,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction< plus, zero,minus>())) this->loop<direction< plus, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< plus, zero, zero>())) this->loop<direction< plus, zero, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< plus, zero, plus>())) this->loop<direction< plus, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                                                                    \
            if (predicate(direction< plus, plus,minus>())) this->loop<direction< plus, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< plus, plus, zero>())) this->loop<direction< plus, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            if (predicate(direction< plus, plus, plus>())) this->loop<direction< plus, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
        }

        BOOST_PP_REPEAT(GT_MAX_ARGS, GTAPPLY, _)
    };

#undef GTLOOP
#undef GTAPPLY

} // namespace gridtools
