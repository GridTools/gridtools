#pragma once

#include "../common/defs.h"

namespace gridtools {
    /**
       Enum defining the directions in a discrete Cartesian grid
     */
    enum sign {any=-2, minus=-1, zero, plus};

    /**
       Class defining a direction in a cartesian 3D grid.
     */
    template <sign I_, sign J_, sign K_>
    struct direction {
        static const sign I = I_;
        static const sign J = J_;
        static const sign K = K_;
    };


    /**
       Struct to apply user specified boundary condition cases on data fields.

       \tparam BoundaryFunction The user class defining the operations on the boundary. It must be copy constructible.
       \tparam HaloDescriptors  The type behaving as a read only array of halo descriptors
     */
    template <typename BoundaryFunction, typename HaloDescriptors = array<halo_descriptor, 3> >
    struct boundary_apply {
    private:
        HaloDescriptors halo_descriptors;
        BoundaryFunction const boundary_function;

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

        BOOST_PP_REPEAT(2, GTLOOP, _)

        public:
        boundary_apply(HaloDescriptors const& hd)
            : halo_descriptors(hd)
            , boundary_function(BoundaryFunction())
        {}

        boundary_apply(HaloDescriptors const& hd, BoundaryFunction const & bf)
            : halo_descriptors(hd)
            , boundary_function(bf)
        {}

#define GTAPPLY(z, n, nil)                                                \
        template <BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), typename DataField)> \
        void apply(BOOST_PP_ENUM_BINARY_PARAMS(BOOST_PP_INC(n), DataField, & data_field) ) const { \
                                                                        \
            this->loop<direction<minus,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<minus,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<minus,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<minus, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<minus, zero, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<minus, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<minus, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<minus, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<minus, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<zero,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<zero,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<zero,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<zero, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<zero, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<zero, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<zero, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<zero, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<plus,minus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<plus,minus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<plus,minus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<plus, zero,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<plus, zero, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<plus, zero, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
                                                                        \
            this->loop<direction<plus, plus,minus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<plus, plus, zero> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
            this->loop<direction<plus, plus, plus> >(BOOST_PP_ENUM_PARAMS(BOOST_PP_INC(n), data_field)); \
        }

        BOOST_PP_REPEAT(2, GTAPPLY, _)
    };

} // namespace gridtools
