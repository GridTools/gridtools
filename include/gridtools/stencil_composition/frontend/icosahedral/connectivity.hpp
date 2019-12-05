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

#include "../../../common/array.hpp"
#include "../../../common/defs.hpp"
#include "location_type.hpp"

namespace gridtools {
    namespace icosahedral {

        // static triple dispatch
        template <class From>
        struct from {
            template <class To>
            struct to {
                template <uint_t Color>
                struct with_color;
            };
        };

        template <size_t N>
        using position_offsets_type = array<array<int_t, 4>, N> const;

        /**
         * Following specializations provide all information about the connectivity of the icosahedral/ocahedral grid
         * While ordering is arbitrary up to some extent, if must respect some rules that user expect, and that conform
         * part of an API. Rules are the following:
         *   1. Flow variables on edges by convention are outward on downward cells (color 0) and inward on upward cells
         * (color 1)
         *      as depicted below
         @verbatim
                  ^
                  |                   /\
             _____|____              /  \
             \        /             /    \
              \      /             /      \
          <----\    /---->        /-->  <--\
                \  /             /     ^    \
                 \/             /______|_____\
         @endverbatim
         *   2. Neighbor edges of a cell must follow the same convention than neighbor cells of a cell. I.e. the
         following
         *
         @verbatim
                 /\
                1  2
               /_0__\
           imposes
              ____________
              \    /\    /
               \1 /  \2 /
                \/____\/
                 \  0 /
                  \  /
                   \/
         @endverbatim
         *
         *   3. Cell neighbours of an edge, in the order 0 -> 1 follow the direction of the flow (N_t) on edges defined
         in
         * 1.
         *      This fixes the order of cell neighbors of an edge
         *
         *   4. Vertex neighbors of an edge, in the order 0 -> 1 defines a vector N_l which is perpendicular to N_t.
         *      This fixes the order of vertex neighbors of an edge
         *
         */
        template <>
        template <>
        template <>
        struct from<cells>::to<cells>::with_color<1> {
            /*
             * neighbors order
             *
             @verbatim
               ____________
               \    /\    /
                \1 /  \2 /
                 \/____\/
                  \  0 /
                   \  /
                    \/
             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<3> offsets() {
                return {{
                    {+1, +0, 0, -1}, //
                    {+0, +0, 0, -1}, //
                    {+0, +1, 0, -1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<cells>::to<cells>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim
                     /\
                    /0 \
                   /____\
                  /\    /\
                 /2 \  /1 \
                /____\/____\
             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<3> offsets() {
                return {{
                    {-1, +0, 0, +1}, //
                    {+0, +0, 0, +1}, //
                    {+0, -1, 0, +1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<vertices>::to<vertices>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim

                    1____2
                   /\    /\
                  /  \  /  \
                 0____\/____3
                 \    /\    /
                  \  /  \  /
                   \5____4/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<6> offsets() {
                return {{
                    {+0, -1, 0, +0}, //
                    {-1, +0, 0, +0}, //
                    {-1, +1, 0, +0}, //
                    {+0, +1, 0, +0}, //
                    {+1, +0, 0, +0}, //
                    {+1, -1, 0, +0}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<edges>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim

                   __1___
                  /\    /
                 0  \  2
                /_3__\/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<4> offsets() {
                return {{
                    {+0, -1, 0, +2}, //
                    {+0, +0, 0, +1}, //
                    {+0, +0, 0, +2}, //
                    {+1, -1, 0, +1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<edges>::with_color<1> {
            /*
             * neighbors order
             *
             @verbatim

                 /\
                0  1
               /____\
               \    /
                3  2
                 \/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<4> offsets() {
                return {{
                    {-1, +0, 0, +1}, //
                    {-1, +1, 0, -1}, //
                    {+0, +0, 0, +1}, //
                    {+0, +0, 0, -1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<edges>::with_color<2> {
            /*
             * neighbors order
             *
             @verbatim

               __1___
               \    /\
                0  /  2
                 \/_3__\

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<4> offsets() {
                return {{
                    {+0, +0, 0, -2}, //
                    {+0, +0, 0, -1}, //
                    {+0, +1, 0, -2}, //
                    {+1, +0, 0, -1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<cells>::to<edges>::with_color<1> {
            /*
             * neighbors order
             *
             @verbatim

                  /\
                 1  2
                /_0__\

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<3> offsets() {
                return {{
                    {+1, +0, 0, +0}, //
                    {+0, +0, 0, +1}, //
                    {+0, +1, 0, -1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<cells>::to<edges>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim

               __0___
               \    /
                2  1
                 \/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<3> offsets() {
                return {{
                    {+0, +0, 0, +1}, //
                    {+0, +0, 0, +2}, //
                    {+0, +0, 0, +0}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<cells>::to<vertices>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim

              1______2
               \    /
                \  /
                 \/
                 0

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<3> offsets() {
                return {{
                    {+1, +0, 0, +0}, //
                    {+0, +0, 0, +0}, //
                    {+0, +1, 0, +0}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<cells>::to<vertices>::with_color<1> {
            /*
             * neighbors order
             *
             @verbatim

                  0
                  /\
                 /  \
                /____\
               2      1

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<3> offsets() {
                return {{
                    {+0, +1, 0, -1}, //
                    {+1, +1, 0, -1}, //
                    {+1, +0, 0, -1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<cells>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim
                   ______
                  /\  1 /
                 /0 \  /
                /____\/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<2> offsets() {
                return {{
                    {+0, -1, 0, +1}, //
                    {+0, +0, 0, +0}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<cells>::with_color<1> {
            /*
             * neighbors order
             *
             @verbatim

                  /\
                 / 0\
                /____\
                \    /
                 \1 /
                  \/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<2> offsets() {
                return {{
                    {-1, +0, 0, +0}, //
                    {+0, +0, 0, -1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<cells>::with_color<2> {
            /*
             * neighbors order
             *
             @verbatim
               ______
               \ 1  /\
                \  / 0\
                 \/____\

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<2> offsets() {
                return {{
                    {+0, +0, 0, -1}, //
                    {+0, +0, 0, -2}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<vertices>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim

                  1______
                  /\    /
                 /  \  /
                /____\/
                      0

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<2> offsets() {
                return {{
                    {+1, +0, 0, +0}, //
                    {+0, +0, 0, +0}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<vertices>::with_color<1> {
            /*
             * neighbors order
             *
             @verbatim

                  /\
                 /  \
               0/____\1
                \    /
                 \  /
                  \/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<2> offsets() {
                return {{
                    {+0, +0, 0, -1}, //
                    {+0, +1, 0, -1}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<edges>::to<vertices>::with_color<2> {
            /*
             * neighbors order
             *
             @verbatim

               ______0
               \    /\
                \  /  \
                 \/____\
                 1

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<2> offsets() {
                return {{
                    {+0, +1, 0, -2}, //
                    {+1, +0, 0, -2}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<vertices>::to<cells>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim
                   ______
                  /\ 1  /\
                 /0 \  / 2\
                /____\/____\
                \ 5  /\ 3  /
                 \  /4 \  /
                  \/____\/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<6> offsets() {
                return {{
                    {-1, -1, 0, +1}, //
                    {-1, +0, 0, +0}, //
                    {-1, +0, 0, +1}, //
                    {+0, +0, 0, +0}, //
                    {+0, -1, 0, +1}, //
                    {+0, -1, 0, +0}  //
                }};
            }
        };

        template <>
        template <>
        template <>
        struct from<vertices>::to<edges>::with_color<0> {
            /*
             * neighbors order
             *
             @verbatim
                   ______
                  /\    /\
                 /  1  2  \
                /__0_\/__3_\
                \    /\    /
                 \  5  4  /
                  \/____\/

             @endverbatim
             */
            GT_FUNCTION
            constexpr static position_offsets_type<6> offsets() {
                return {{
                    {+0, -1, 0, +1}, //
                    {-1, +0, 0, +0}, //
                    {-1, +0, 0, +2}, //
                    {+0, +0, 0, +1}, //
                    {+0, +0, 0, +0}, //
                    {+0, -1, 0, +2}  //
                }};
            }
        };

        template <typename SrcLocation, typename DestLocation, uint_t Color>
        struct connectivity {

            static_assert(is_location_type<SrcLocation>::value, "Error: unknown src location type");
            static_assert(is_location_type<DestLocation>::value, "Error: unknown dst location type");
            static_assert(Color < SrcLocation::value, "Error: Color index beyond color length");

            GT_FUNCTION constexpr static auto offsets() {
                return from<SrcLocation>::template to<DestLocation>::template with_color<Color>::offsets();
            }
        };
    } // namespace icosahedral
} // namespace gridtools
