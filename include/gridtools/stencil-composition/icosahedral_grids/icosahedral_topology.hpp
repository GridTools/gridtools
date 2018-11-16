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

#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/as_set.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/sequence/intrinsic/at.hpp>
#include <boost/fusion/view/zip_view.hpp>

#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/vector/vector_fwd.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/io.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/vector_fwd.hpp>
#include <boost/fusion/sequence/io.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>

#include "../../common/array.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/generic_metafunctions/is_all_integrals.hpp"
#include "../../common/generic_metafunctions/pack_get_elem.hpp"
#include "../../common/gt_assert.hpp"
#include "../../storage/common/halo.hpp"
#include "../location_type.hpp"
#include "position_offset_type.hpp"

#include "icosahedral_topology_metafunctions.hpp"

namespace gridtools {

    // static triple dispatch
    template <typename Location1>
    struct from {
        template <typename Location2>
        struct to {
            template <typename Color>
            struct with_color;
        };
    };

    template <typename T>
    struct is_grid_topology;

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
     *   2. Neighbor edges of a cell must follow the same convention than neighbor cells of a cell. I.e. the following
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
     *   3. Cell neighbours of an edge, in the order 0 -> 1 follow the direction of the flow (N_t) on edges defined in
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
    struct from<enumtype::cells>::to<enumtype::cells>::with_color<static_uint<1>> {
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
        constexpr static position_offsets_type<3> offsets() { return {{{1, -1, 0, 0}, {0, -1, 0, 0}, {0, -1, 1, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::cells>::to<enumtype::cells>::with_color<static_uint<0>> {
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
        constexpr static position_offsets_type<3> offsets() { return {{{-1, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, -1, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::vertices>::to<enumtype::vertices>::with_color<static_uint<0>> {
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
            return {{{0, 0, -1, 0}, {-1, 0, 0, 0}, {-1, 0, 1, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::edges>::with_color<static_uint<0>> {
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
            return {{{0, 2, -1, 0}, {0, 1, 0, 0}, {0, 2, 0, 0}, {1, 1, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::edges>::with_color<static_uint<1>> {
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
            return {{{-1, 1, 0, 0}, {-1, -1, 1, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::edges>::with_color<static_uint<2>> {
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
            return {{{0, -2, 0, 0}, {0, -1, 0, 0}, {0, -2, 1, 0}, {1, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::cells>::to<enumtype::edges>::with_color<static_uint<1>> {
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
        constexpr static position_offsets_type<3> offsets() { return {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 1, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::cells>::to<enumtype::edges>::with_color<static_uint<0>> {
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
        constexpr static position_offsets_type<3> offsets() { return {{{0, 1, 0, 0}, {0, 2, 0, 0}, {0, 0, 0, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::cells>::to<enumtype::vertices>::with_color<static_uint<0>> {
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
        constexpr static position_offsets_type<3> offsets() { return {{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::cells>::to<enumtype::vertices>::with_color<static_uint<1>> {
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
        constexpr static position_offsets_type<3> offsets() { return {{{0, -1, 1, 0}, {1, -1, 1, 0}, {1, -1, 0, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::cells>::with_color<static_uint<0>> {
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
        constexpr static position_offsets_type<2> offsets() { return {{{0, 1, -1, 0}, {0, 0, 0, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::cells>::with_color<static_uint<1>> {
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
        constexpr static position_offsets_type<2> offsets() { return {{{-1, 0, 0, 0}, {0, -1, 0, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::cells>::with_color<static_uint<2>> {
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
        constexpr static position_offsets_type<2> offsets() { return {{{0, -1, 0, 0}, {0, -2, 0, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::vertices>::with_color<static_uint<0>> {
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
        constexpr static position_offsets_type<2> offsets() { return {{{1, 0, 0, 0}, {0, 0, 0, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::vertices>::with_color<static_uint<1>> {
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
        constexpr static position_offsets_type<2> offsets() { return {{{0, -1, 0, 0}, {0, -1, 1, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::edges>::to<enumtype::vertices>::with_color<static_uint<2>> {
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
        constexpr static position_offsets_type<2> offsets() { return {{{0, -2, 1, 0}, {1, -2, 0, 0}}}; }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::vertices>::to<enumtype::cells>::with_color<static_uint<0>> {
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
            return {{{-1, 1, -1, 0}, {-1, 0, 0, 0}, {-1, 1, 0, 0}, {0, 0, 0, 0}, {0, 1, -1, 0}, {0, 0, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from<enumtype::vertices>::to<enumtype::edges>::with_color<static_uint<0>> {
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
            return {{{0, 1, -1, 0}, {-1, 0, 0, 0}, {-1, 2, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 2, -1, 0}}};
        }
    };

    template <typename SrcLocation, typename DestLocation, uint_t Color>
    struct connectivity {

        GRIDTOOLS_STATIC_ASSERT((is_location_type<SrcLocation>::value), "Error: unknown src location type");
        GRIDTOOLS_STATIC_ASSERT((is_location_type<DestLocation>::value), "Error: unknown dst location type");
        GRIDTOOLS_STATIC_ASSERT(Color < SrcLocation::n_colors::value, "Error: Color index beyond color length");

        GT_FUNCTION constexpr static auto offsets() GT_AUTO_RETURN(
            from<SrcLocation>::template to<DestLocation>::template with_color<static_uint<Color>>::offsets());
    };

    /**
     */
    template <typename Backend>
    class icosahedral_topology {
      public:
        using cells = enumtype::cells;
        using edges = enumtype::edges;
        using vertices = enumtype::vertices;
        using type = icosahedral_topology<Backend>;

        // returns a layout map with ordering specified by the Backend but where
        // the user can specify the active dimensions
        template <typename Selector>
        using layout_t = typename Backend::template select_layout<Selector>::type;

        template <typename LocationType, typename Halo = halo<0, 0, 0, 0>, typename Selector = selector<1, 1, 1, 1>>
        using meta_storage_t = typename Backend::
            template storage_info_t<impl::compute_uuid<LocationType::value, Selector>::value, layout_t<Selector>, Halo>;

        template <typename LocationType,
            typename ValueType,
            typename Halo = halo<0, 0, 0, 0>,
            typename Selector = selector<1, 1, 1, 1>>
        using data_store_t =
            typename Backend::template data_store_t<ValueType, meta_storage_t<LocationType, Halo, Selector>>;

        array<uint_t, 3> m_dims; // Sizes as cells in a multi-dimensional Cell array

      public:
        template <typename... UInt>

        GT_FUNCTION icosahedral_topology(uint_t idim, uint_t jdim, uint_t kdim) : m_dims{idim, jdim, kdim} {}

        template <typename LocationType,
            typename ValueType,
            typename Halo = halo<0, 0, 0, 0>,
            typename Selector = selector<1, 1, 1, 1>,
            typename... IntTypes,
            typename std::enable_if<is_all_integral<IntTypes...>::value, int>::type = 0>
        data_store_t<LocationType, ValueType, Halo, Selector> make_storage(
            char const *name, IntTypes... extra_dims) const {
            GRIDTOOLS_STATIC_ASSERT(is_location_type<LocationType>::value, "ERROR: location type is wrong");
            GRIDTOOLS_STATIC_ASSERT(is_selector<Selector>::value, "ERROR: dimension selector is wrong");
            GRIDTOOLS_STATIC_ASSERT(
                Selector::size == sizeof...(IntTypes) + 4, "ERROR: Mismatch between Selector and extra-dimensions");

            using meta_storage_type = meta_storage_t<LocationType, Halo, Selector>;
            GRIDTOOLS_STATIC_ASSERT(Selector::size == meta_storage_type::layout_t::masked_length,
                "ERROR: Mismatch between Selector and space dimensions");

            return {{m_dims[0], LocationType::n_colors::value, m_dims[1], m_dims[2], extra_dims...}, name};
        }
    };

    template <typename T>
    struct is_grid_topology : std::false_type {};

    template <typename Backend>
    struct is_grid_topology<icosahedral_topology<Backend>> : std::true_type {};

} // namespace gridtools
