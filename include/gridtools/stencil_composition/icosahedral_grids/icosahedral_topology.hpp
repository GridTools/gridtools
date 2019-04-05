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

#include "../../common/array.hpp"
#include "../../common/generic_metafunctions/is_all_integrals.hpp"
#include "../../common/generic_metafunctions/pack_get_elem.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/layout_map_metafunctions.hpp"
#include "../../meta/iseq_to_list.hpp"
#include "../../meta/list_to_iseq.hpp"
#include "../../meta/take.hpp"
#include "../../storage/common/halo.hpp"
#include "../../storage/storage_facility.hpp"
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

        GT_STATIC_ASSERT((is_location_type<SrcLocation>::value), "Error: unknown src location type");
        GT_STATIC_ASSERT((is_location_type<DestLocation>::value), "Error: unknown dst location type");
        GT_STATIC_ASSERT(Color < SrcLocation::n_colors::value, "Error: Color index beyond color length");

        GT_FUNCTION constexpr static auto offsets() GT_AUTO_RETURN(
            from<SrcLocation>::template to<DestLocation>::template with_color<static_uint<Color>>::offsets());
    };

    namespace _impl {
        template <class>
        struct default_layout;
        template <>
        struct default_layout<backend::cuda> {
            using type = layout_map<3, 2, 1, 0>;
        };
        template <>
        struct default_layout<backend::x86> {
            using type = layout_map<0, 1, 2, 3>;
        };
        template <>
        struct default_layout<backend::naive> {
            using type = layout_map<0, 1, 2, 3>;
        };

        template <std::size_t N, class DimSelector>
        using shorten_selector = GT_META_CALL(
            meta::list_to_iseq, (GT_META_CALL(meta::take_c, (N, GT_META_CALL(meta::iseq_to_list, DimSelector)))));
    } // namespace _impl

    /**
     */
    template <typename Backend>
    class icosahedral_topology {
      private:
        template <typename DimSelector>
        struct select_layout {
            using layout_map_t = typename _impl::default_layout<Backend>::type;
            using dim_selector_4d_t = _impl::shorten_selector<4, DimSelector>;
            using filtered_layout = typename get_special_layout<layout_map_t, dim_selector_4d_t>::type;

            using type = typename conditional_t<(DimSelector::size() > 4),
                extend_layout_map<filtered_layout, DimSelector::size() - 4>,
                meta::lazy::id<filtered_layout>>::type;
        };

      public:
        using cells = enumtype::cells;
        using edges = enumtype::edges;
        using vertices = enumtype::vertices;
        using type = icosahedral_topology<Backend>;

        // returns a layout map with ordering specified by the Backend but where
        // the user can specify the active dimensions
        template <typename Selector>
        using layout_t = typename select_layout<Selector>::type;

        template <typename LocationType, typename Halo = halo<0, 0, 0, 0>, typename Selector = selector<1, 1, 1, 1>>
        using meta_storage_t = typename storage_traits<Backend>::template custom_layout_storage_info_t<
            impl::compute_uuid<LocationType::value, Selector>::value,
            layout_t<Selector>,
            Halo>;

        template <typename LocationType,
            typename ValueType,
            typename Halo = halo<0, 0, 0, 0>,
            typename Selector = selector<1, 1, 1, 1>>
        using data_store_t = typename storage_traits<Backend>::template data_store_t<ValueType,
            meta_storage_t<LocationType, Halo, Selector>>;

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
            GT_STATIC_ASSERT(is_location_type<LocationType>::value, "ERROR: location type is wrong");
            GT_STATIC_ASSERT(is_selector<Selector>::value, "ERROR: dimension selector is wrong");
            GT_STATIC_ASSERT(
                Selector::size() == sizeof...(IntTypes) + 4, "ERROR: Mismatch between Selector and extra-dimensions");

            using meta_storage_type = meta_storage_t<LocationType, Halo, Selector>;
            GT_STATIC_ASSERT(Selector::size() == meta_storage_type::layout_t::masked_length,
                "ERROR: Mismatch between Selector and space dimensions");

            return {{m_dims[0], LocationType::n_colors::value, m_dims[1], m_dims[2], extra_dims...}, name};
        }
    };

    template <typename T>
    struct is_grid_topology : std::false_type {};

    template <typename Backend>
    struct is_grid_topology<icosahedral_topology<Backend>> : std::true_type {};

} // namespace gridtools
