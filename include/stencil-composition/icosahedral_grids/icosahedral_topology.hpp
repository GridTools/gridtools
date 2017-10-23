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
#include <boost/fusion/sequence/intrinsic/at.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/as_set.hpp>

#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/container/vector/vector_fwd.hpp>
#include <boost/fusion/include/vector_fwd.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/sequence/io.hpp>
#include <boost/fusion/include/io.hpp>

#include "../../common/array.hpp"
#include "../../common/gt_assert.hpp"
#include "../location_type.hpp"
#include "../../common/array_addons.hpp"
#include "../../common/gpu_clone.hpp"
#include "../../common/generic_metafunctions/pack_get_elem.hpp"
#include "../../common/generic_metafunctions/all_integrals.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"

#include "icosahedral_topology_metafunctions.hpp"

namespace gridtools {

    // TODO this is duplicated below

    namespace {
        using cells = location_type< 0, 2 >;
        using edges = location_type< 1, 3 >;
        using vertices = location_type< 2, 1 >;
    }

    namespace impl {
        template < typename StorageInfo, typename Array, unsigned N = Array::n_dimensions, typename... Rest >
        constexpr typename boost::enable_if_c< (N == 0), StorageInfo >::type get_storage_info_from_array(
            Array arr, Rest... r) {
            GRIDTOOLS_STATIC_ASSERT(
                is_array< Array >::value, GT_INTERNAL_ERROR_MSG("Passed type is not an array type."));
            return StorageInfo(r...);
        }

        template < typename StorageInfo, typename Array, unsigned N = Array::n_dimensions, typename... Rest >
        constexpr typename boost::enable_if_c< (N > 0), StorageInfo >::type get_storage_info_from_array(
            Array arr, Rest... r) {
            GRIDTOOLS_STATIC_ASSERT(
                is_array< Array >::value, GT_INTERNAL_ERROR_MSG("Passed type is not an array type."));
            typedef typename StorageInfo::halo_t halo_t;
            return get_storage_info_from_array< StorageInfo, Array, N - 1 >(arr, arr[N - 1], r...);
        }
    }

    template < typename T, typename ValueType >
    struct return_type {
        typedef array< ValueType, 0 > type;
    };

    // static triple dispatch
    template < typename Location1 >
    struct from {
        template < typename Location2 >
        struct to {
            template < typename Color >
            struct with_color;
        };
    };

    template < typename ValueType >
    struct return_type< from< cells >::template to< cells >, ValueType > {
        typedef array< ValueType, 3 > type;
    };

    template < typename ValueType >
    struct return_type< from< cells >::template to< edges >, ValueType > {
        typedef array< ValueType, 3 > type;
    };

    template < typename ValueType >
    struct return_type< from< cells >::template to< vertices >, ValueType > {
        typedef array< ValueType, 3 > type;
    };

    template < typename ValueType >
    struct return_type< from< edges >::template to< edges >, ValueType > {
        typedef array< ValueType, 4 > type;
    };

    template < typename ValueType >
    struct return_type< from< edges >::template to< cells >, ValueType > {
        typedef array< ValueType, 2 > type;
    };

    template < typename ValueType >
    struct return_type< from< edges >::template to< vertices >, ValueType > {
        typedef array< ValueType, 2 > type;
    };

    template < typename ValueType >
    struct return_type< from< vertices >::template to< vertices >, ValueType > {
        typedef array< ValueType, 6 > type;
    };

    template < typename ValueType >
    struct return_type< from< vertices >::template to< cells >, ValueType > {
        typedef array< ValueType, 6 > type;
    };

    template < typename ValueType >
    struct return_type< from< vertices >::template to< edges >, ValueType > {
        typedef array< ValueType, 6 > type;
    };

    template < uint_t SourceColor >
    struct get_connectivity_offset {

        template < int Idx >
        struct get_element {
            GT_FUNCTION
            constexpr get_element() {}

            template < typename Offsets >
            GT_FUNCTION constexpr static array< uint_t, 4 > apply(array< uint_t, 3 > const &i, Offsets offsets) {
                return {i[0] + offsets[Idx][0],
                    SourceColor + offsets[Idx][1],
                    i[1] + offsets[Idx][2],
                    i[2] + offsets[Idx][3]};
            }
        };
    };

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
    struct from< cells >::to< cells >::with_color< static_uint< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< cells >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, -1, 0, 0}, {0, -1, 0, 0}, {0, -1, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< cells >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< cells >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{-1, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< vertices >::to< vertices >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< vertices >::to< vertices >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{
                {{0, 0, -1, 0}, {-1, 0, 0, 0}, {-1, 0, 1, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< edges >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< edges >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, 2, -1, 0}, {0, 1, 0, 0}, {0, 2, 0, 0}, {1, 1, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< edges >::with_color< static_uint< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< edges >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{-1, 1, 0, 0}, {-1, -1, 1, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< edges >::with_color< static_uint< 2 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< edges >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -2, 0, 0}, {0, -1, 0, 0}, {0, -2, 1, 0}, {1, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< edges >::with_color< static_uint< 1 > > {

        template < typename ValueType = int_t >
        using return_t = typename return_type< from< cells >::to< edges >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< edges >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< edges >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, 1, 0, 0}, {0, 2, 0, 0}, {0, 0, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< vertices >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< vertices >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< cells >::to< vertices >::with_color< static_uint< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< cells >::to< vertices >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -1, 1, 0}, {1, -1, 1, 0}, {1, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< cells >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< cells >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, 1, -1, 0}, {0, 0, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< cells >::with_color< static_uint< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< cells >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{-1, 0, 0, 0}, {0, -1, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< cells >::with_color< static_uint< 2 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< cells >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -1, 0, 0}, {0, -2, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< vertices >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< vertices >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{1, 0, 0, 0}, {0, 0, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< vertices >::with_color< static_uint< 1 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< vertices >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -1, 0, 0}, {0, -1, 1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< edges >::to< vertices >::with_color< static_uint< 2 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< edges >::to< vertices >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{{{0, -2, 1, 0}, {1, -2, 0, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< vertices >::to< cells >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< vertices >::to< cells >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{
                {{-1, 1, -1, 0}, {-1, 0, 0, 0}, {-1, 1, 0, 0}, {0, 0, 0, 0}, {0, 1, -1, 0}, {0, 0, -1, 0}}};
        }
    };

    template <>
    template <>
    template <>
    struct from< vertices >::to< edges >::with_color< static_uint< 0 > > {

        template < typename ValueType >
        using return_t = typename return_type< from< vertices >::to< edges >, ValueType >::type;

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
        constexpr static return_t< array< int_t, 4 > > offsets() {
            return return_t< array< int_t, 4 > >{
                {{0, 1, -1, 0}, {-1, 0, 0, 0}, {-1, 2, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 2, -1, 0}}};
        }
    };

    template < typename SrcLocation, typename DestLocation, uint_t Color >
    struct connectivity {

        GRIDTOOLS_STATIC_ASSERT((is_location_type< SrcLocation >::value), "Error: unknown src location type");
        GRIDTOOLS_STATIC_ASSERT((is_location_type< DestLocation >::value), "Error: unknown dst location type");

        GRIDTOOLS_STATIC_ASSERT(
            (!boost::is_same< SrcLocation, cells >::value || Color < 2), "Error: Color index beyond color length");
        GRIDTOOLS_STATIC_ASSERT(
            (!boost::is_same< SrcLocation, edges >::value || Color < 3), "Error: Color index beyond color length");
        GRIDTOOLS_STATIC_ASSERT(
            (!boost::is_same< SrcLocation, vertices >::value || Color < 1), "Error: Color index beyond color length");

        GT_FUNCTION
        constexpr static
            typename return_type< typename from< SrcLocation >::template to< DestLocation >, array< int_t, 4 > >::type
            offsets() {
            return from< SrcLocation >::template to< DestLocation >::template with_color<
                static_uint< Color > >::offsets();
        }
    };

    /**
    */
    template < typename Backend >
    class icosahedral_topology {
      public:
        using cells = enumtype::cells;
        using edges = enumtype::edges;
        using vertices = enumtype::vertices;
        // default 4d layout map (matching the chosen architecture by the Backend)
        using default_4d_layout_map_t = typename Backend::layout_map_t;
        using type = icosahedral_topology< Backend >;

        // returns a layout map with ordering specified by the Backend but where
        // the user can specify the active dimensions
        template < typename Selector >
        using layout_t = typename Backend::template select_layout< Selector >::type;

        template < typename LocationType,
            typename Halo = halo< 0, 0, 0, 0 >,
            typename Selector = selector< 1, 1, 1, 1 > >
        using meta_storage_t =
            typename Backend::template storage_info_t< impl::compute_uuid< LocationType::value, Selector >::value,
                layout_t< Selector >,
                Halo >;

        template < typename LocationType,
            typename ValueType,
            typename Halo = halo< 0, 0, 0, 0 >,
            typename Selector = selector< 1, 1, 1, 1 > >
        using storage_t =
            typename Backend::template storage_t< ValueType, meta_storage_t< LocationType, Halo, Selector > >;

        const array< uint_t, 3 > m_dims; // Sizes as cells in a multi-dimensional Cell array

      public:
        icosahedral_topology() = delete;

        template < typename... UInt >
        GT_FUNCTION icosahedral_topology(uint_t idim, uint_t jdim, uint_t kdim)
            : m_dims{idim, jdim, kdim} {}

        template < typename LocationType,
            typename ValueType,
            typename Halo = halo< 0, 0, 0, 0 >,
            typename Selector = selector< 1, 1, 1, 1 >,
            typename... IntTypes
#if defined(CUDA8) || !defined(__CUDACC__)
            ,
            typename Dummy = all_integers< IntTypes... >
#endif
            >
        storage_t< LocationType, ValueType, Halo, Selector > make_storage(
            char const *name, IntTypes... extra_dims) const {
            GRIDTOOLS_STATIC_ASSERT((is_location_type< LocationType >::value), "ERROR: location type is wrong");
            GRIDTOOLS_STATIC_ASSERT((is_selector< Selector >::value), "ERROR: dimension selector is wrong");

            GRIDTOOLS_STATIC_ASSERT(
                (Selector::size == sizeof...(IntTypes) + 4), "ERROR: Mismatch between Selector and extra-dimensions");

            using meta_storage_type = meta_storage_t< LocationType, Halo, Selector >;
            GRIDTOOLS_STATIC_ASSERT((Selector::size == meta_storage_type::layout_t::masked_length),
                "ERROR: Mismatch between Selector and space dimensions");

            array< uint_t, meta_storage_type::layout_t::masked_length > metastorage_sizes =
                impl::array_dim_initializers< uint_t,
                    meta_storage_type::layout_t::masked_length,
                    LocationType,
                    Selector >::apply(m_dims, extra_dims...);
            auto ameta = impl::get_storage_info_from_array< meta_storage_type >(metastorage_sizes);
            return storage_t< LocationType, ValueType, Halo, Selector >(ameta, name);
        }

        template < typename LocationType >
        GT_FUNCTION array< int_t, 4 > ll_indices(array< int_t, 3 > const &i, LocationType) const {
            auto out = array< int_t, 4 >{i[0],
                i[1] % static_cast< int_t >(LocationType::n_colors::value),
                i[1] / static_cast< int >(LocationType::n_colors::value),
                i[2]};
            return array< int_t, 4 >{i[0],
                i[1] % static_cast< int_t >(LocationType::n_colors::value),
                i[1] / static_cast< int >(LocationType::n_colors::value),
                i[2]};
        }
    };
} // namespace gridtools
