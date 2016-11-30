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

#include "common/data_field_view.hpp"
#include "common/data_view.hpp"

#include "../common/pointer.hpp"
#include "arg.hpp"

namespace gridtools {

    template < typename T >
    struct storage_wrapper_base;

    template < typename Arg, typename View, typename TileI, typename TileJ >
    struct storage_wrapper;

    template < typename Arg, typename Storage, bool ReadOnly, typename TileI, typename TileJ >
    struct storage_wrapper< Arg, data_view< Storage, ReadOnly >, TileI, TileJ >
        : storage_wrapper_base< storage_wrapper< Arg, data_view< Storage, ReadOnly >, TileI, TileJ > > {
        static_assert(is_data_store< typename Arg::storage_t >::value,
            "cannot create a (data_store) storage wrapper without a data_store type.");
        using view_t = data_view< Storage, ReadOnly >;
        using super = storage_wrapper_base< storage_wrapper< Arg, view_t, TileI, TileJ > >;
        using super::super;
        constexpr static uint_t storage_size = 1;
        // tell me how I should initialize the ptr_t member called m_data_ptrs
        void initialize(view_t v) { this->m_data_ptrs = v.m_raw_ptr; }

        template < typename T >
        void assign(T &d) const {
            d = this->m_data_ptrs;
        }

        void info() const {
            std::cout << "DataView: \n";
            std::cout << "\tstorage_size: " << storage_size << "\n";
            std::cout << "\t\tptr: " << this->m_data_ptrs << "\n";
            std::cout << "\n";
        }
    };

    template < typename Arg, typename Storage, bool ReadOnly, typename TileI, typename TileJ >
    struct storage_wrapper< Arg, data_field_view< Storage, ReadOnly >, TileI, TileJ >
        : storage_wrapper_base< storage_wrapper< Arg, data_field_view< Storage, ReadOnly >, TileI, TileJ > > {
        static_assert(is_data_store_field< typename Arg::storage_t >::value,
            "cannot create a (data_store_field) storage wrapper without a data_store_field type.");
        using view_t = data_field_view< Storage, ReadOnly >;
        using super = storage_wrapper_base< storage_wrapper< Arg, view_t, TileI, TileJ > >;
        using super::super;
        constexpr static uint_t storage_size = view_t::N;
        // tell me how I should initialize the ptr_t member called m_data_ptrs
        void initialize(view_t v) {
            for (unsigned i = 0; i < view_t::N; ++i)
                this->m_data_ptrs[i] = v.m_raw_ptrs[i];
        }

        template < typename T >
        void assign(T &d) const {
            std::copy(this->m_data_ptrs, this->m_data_ptrs + storage_size, d);
        }

        void info() const {
            std::cout << "DataFieldView: \n";
            std::cout << "\tstorage_size: " << storage_size << "\n";
            for (unsigned i = 0; i < storage_size; ++i) {
                std::cout << "\t\tptr: " << this->m_data_ptrs[i] << "\n";
            }
            std::cout << "\n";
        }
    };

    template < typename Arg, typename View, typename TileI, typename TileJ >
    struct storage_wrapper_base< storage_wrapper< Arg, View, TileI, TileJ > > {
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
        using data_ptr_t = typename view_t::data_ptr_t;

        // some more information
        constexpr static bool is_temporary = arg_t::is_temporary;
        constexpr static bool is_read_only = view_t::ReadOnly;

        // data ptrs
        data_ptr_t m_data_ptrs;
    };

    template < typename T >
    struct is_storage_wrapper : boost::mpl::false_ {};

    template < typename Arg, typename View, typename TileI, typename TileJ >
    struct is_storage_wrapper< storage_wrapper< Arg, View, TileI, TileJ > > : boost::mpl::true_ {};

    template < typename T >
    struct get_arg_from_storage_wrapper {
        typedef typename T::arg_t type;
    };

    template < typename T >
    struct get_storage_size_from_storage_wrapper {
        typedef boost::mpl::int_< T::storage_size > type;
        static const int value = T::storage_size;
    };

    template < typename T >
    struct get_storage_info_from_storage_wrapper {
        typedef typename T::storage_info_t type;
    };

    template < typename T >
    struct get_storage_from_storage_wrapper {
        typedef typename T::storage_t type;
    };

    template < typename T >
    struct get_data_ptr_from_storage_wrapper {
        typedef typename T::data_ptr_t type;
    };

    template < typename T >
    struct get_arg_index_from_storage_wrapper : T::index_t {};

    template < typename EsfArg, typename StorageWrapperList >
    struct get_storage_wrapper_elem {
        typedef typename boost::mpl::fold< StorageWrapperList,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, get_arg_from_storage_wrapper< boost::mpl::_2 > > >::type ArgVec;
        typedef typename boost::mpl::at_c< StorageWrapperList,
            boost::mpl::find< ArgVec, EsfArg >::type::pos::value >::type type;
    };
}