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

#include <boost/mpl/max_element.hpp>
#include <boost/mpl/min_element.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/mpl/filter_view.hpp>

#include "storage-facility.hpp"

#include "../common/pointer.hpp"
#include "arg.hpp"
#include "tile.hpp"

namespace gridtools {

    /**
     * @brief The StorageWrapper class is used to keep together information about storages (data_store,
     * data_store_fields)
     * that are mapped to an arg and a corresponding view type. The contained information is a collection of types
     * (view type, arg type, storage type, data type, storage_info type, etc.) and information about temporary, size
     * read_only, etc.
     * @tparam Arg arg type
     * @tparam View view type associated to the arg type
     * @tparam TileI tiling information in I direction (important for temporaries)
     * @tparam TileJ tiling information in J direction (important for temporaries)
     */
    template < typename Arg, typename View, typename TileI, typename TileJ >
    struct storage_wrapper {
        // checks
        GRIDTOOLS_STATIC_ASSERT((is_arg< Arg >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_data_view< View >::value || is_data_field_view< View >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_tile< TileI >::value && is_tile< TileJ >::value), GT_INTERNAL_ERROR);

        // some type information
        using derived_t = storage_wrapper< Arg, View, TileI, TileJ >;
        using arg_t = Arg;
        using view_t = View;
        using tileI_t = TileI;
        using tileJ_t = TileJ;
        using index_t = typename arg_t::index_t;
        using storage_t = typename arg_t::storage_t;
        using data_t = typename storage_t::data_t;
        using storage_info_t = typename storage_t::storage_info_t;

        // some more information
        constexpr static uint_t storage_size = view_t::view_size;
        constexpr static bool is_temporary = arg_t::is_temporary;
        constexpr static bool is_read_only = (view_t::mode == access_mode::ReadOnly);

        // assign the data ptrs to some other ptrs
        template < typename T >
        void assign(T &d) const {
            std::copy(this->m_data_ptrs, this->m_data_ptrs + storage_size, d);
        }

        // tell me how I should initialize the ptr_t member called m_data_ptrs
        void initialize(view_t v) {
            for (unsigned i = 0; i < storage_size; ++i)
                this->m_data_ptrs[i] = v.m_raw_ptrs[i];
        }

        // data ptrs
        data_t *m_data_ptrs[storage_size];
    };

    /* Storage Wrapper metafunctions */

    template < typename T >
    struct is_storage_wrapper : boost::mpl::false_ {};

    template < typename Arg, typename View, typename TileI, typename TileJ >
    struct is_storage_wrapper< storage_wrapper< Arg, View, TileI, TileJ > > : boost::mpl::true_ {};

    template < typename T >
    struct temporary_info_from_storage_wrapper : boost::mpl::bool_< T::is_temporary > {};

    template < typename T >
    struct arg_from_storage_wrapper {
        typedef typename T::arg_t type;
    };

    template < typename T >
    struct storage_size_from_storage_wrapper {
        typedef boost::mpl::int_< T::storage_size > type;
        static const int value = T::storage_size;
    };

    template < typename T >
    struct storage_info_from_storage_wrapper {
        typedef typename T::storage_info_t type;
    };

    template < typename T >
    struct storage_from_storage_wrapper {
        typedef typename T::storage_t type;
    };

    template < typename T >
    struct data_ptr_from_storage_wrapper {
        typedef typename T::data_t *type[T::storage_size];
    };

    template < typename T >
    struct arg_index_from_storage_wrapper : T::index_t {};

    template < typename EsfArg, typename StorageWrapperList >
    struct storage_wrapper_elem {
        typedef typename boost::mpl::fold< StorageWrapperList,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, arg_from_storage_wrapper< boost::mpl::_2 > > >::type ArgVec;
        typedef typename boost::mpl::at_c< StorageWrapperList,
            boost::mpl::find< ArgVec, EsfArg >::type::pos::value >::type type;
    };

    /** @brief get tiling information out of a given storage wrapper.
     *  @tparam Coord coordinate (I --> 0, J --> 1)
     */
    template < unsigned Coord >
    struct tile_from_storage_wrapper;

    template <>
    struct tile_from_storage_wrapper< 0 > {
        template < typename StorageWrapper >
        struct apply {
            typedef typename StorageWrapper::tileI_t type;
        };
    };

    template <>
    struct tile_from_storage_wrapper< 1 > {
        template < typename StorageWrapper >
        struct apply {
            typedef typename StorageWrapper::tileJ_t type;
        };
    };

    /** @brief get the maximum extent in I direction from a given storage wrapper list.
     *  This information is needed when using temporary storages.
     *  @tparam StorageWrapperList given storage wrapper list
     */
    template < typename StorageWrapperList >
    struct max_i_extent_from_storage_wrapper_list {
        typedef
            typename boost::mpl::transform< StorageWrapperList, tile_from_storage_wrapper< 1 > >::type all_i_tiles_t;
        typedef typename boost::mpl::transform< all_i_tiles_t, get_minus_t_from_tile< boost::mpl::_ > >::type
            all_i_minus_tiles_t;
        typedef typename boost::mpl::transform< all_i_tiles_t, get_plus_t_from_tile< boost::mpl::_ > >::type
            all_i_plus_tiles_t;
        typedef typename boost::mpl::deref< typename boost::mpl::max_element< all_i_minus_tiles_t >::type >::type
            min_i_minus_t;
        typedef typename boost::mpl::deref< typename boost::mpl::max_element< all_i_plus_tiles_t >::type >::type
            max_i_plus_t;
        typedef typename boost::mpl::deref<
            typename boost::mpl::max_element< boost::mpl::vector< max_i_plus_t, min_i_minus_t > >::type >::type type;
    };
}
