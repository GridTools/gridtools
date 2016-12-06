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
namespace gridtools {

    template < typename First, typename... StorageExtended >
    struct data_field;

    /** @brief traits class defining some useful compile-time counters
     */
    template < typename First, typename... StorageExtended >
    struct dimension_extension_traits {
        static const bool is_rectangular = accumulate(logical_and(), (StorageExtended::n_width == First::n_width)...);
        // total number of snapshots in the discretized data field
        static const ushort_t n_fields = First::n_width + dimension_extension_traits< StorageExtended... >::n_fields;
        // the buffer size of the current dimension (i.e. the number of snapshots in one dimension)
        static const short_t n_width = First::n_width;
        // the number of dimensions (i.e. the number of different storage_lists)
        static const ushort_t n_dimensions = dimension_extension_traits< StorageExtended... >::n_dimensions + 1;
        // the current field extension
        // n_fields-1 because the storage_list takes the EXTRA width as argument, not the total width.
        typedef storage_list< First, n_fields - 1 > type;
        // typedef First type;
        typedef dimension_extension_traits< StorageExtended... > super;
    };

    /**@brief fallback in case the snapshot we try to access exceeds the width dimension assigned to a discrete scalar
     * field*/
    struct dimension_extension_null {
        static const ushort_t n_fields = 0;
        static const short_t n_width = 0;
        static const ushort_t n_dimensions = 0;
        typedef struct error_index_too_large1 {
        } type;
        typedef struct error_index_too_large2 { } super; };

    /**@brief template specialization at the end of the recustion.*/
    template < typename First >
    struct dimension_extension_traits< First > {
        static constexpr bool is_rectangular = true;
        static const ushort_t n_fields = First::n_width;
        static const short_t n_width = First::n_width;
        static const ushort_t n_dimensions = 1;
        typedef First type;
        typedef dimension_extension_null super;
    };

    template < typename T >
    struct is_dimension_extension_traits : boost::mpl::false_ {};

    template < typename... T >
    struct is_dimension_extension_traits< dimension_extension_traits< T... > > : boost::mpl::true_ {};

    template <>
    struct is_dimension_extension_traits< dimension_extension_null > : boost::mpl::true_ {};

    template < typename T >
    struct get_fields {
        using type = static_int< T::n_fields >;
    };

    template < typename T >
    struct get_value_ {
        using type = static_int< T::value >;
    };

    template < typename T >
    struct get_width {
        using type = static_int< T::n_width >;
    };

    /** @brief metafunction to compute the number of total snapshots present in the data field
        (sum of storage_list::n_width) before
        the ID-th storage list*/
    template < typename Storage, uint_t Id, uint_t IdMax >
    struct compute_storage_offset {

        GRIDTOOLS_STATIC_ASSERT(IdMax >= Id && Id >= 0, "Library internal error");
        typedef typename boost::mpl::eval_if_c< IdMax - Id == 0,
            get_fields< typename Storage::super >,
            get_value_< compute_storage_offset< typename Storage::super, Id + 1, IdMax > > >::type type;
        static const uint_t value = type::value;
    };

    template < typename T >
    struct is_data_field : public boost::mpl::false_ {};

    /** @brief metafunction to compute the number of snapshots present in the ID-th storage_list
        (storage_list::n_width)
    */
    template < typename Storage, uint_t Id, uint_t IdMax >
    struct compute_storage_list_width {
        typedef typename Storage::super next_storage_t;

        GRIDTOOLS_STATIC_ASSERT(IdMax >= Id && Id >= 0, "Library internal error");
        GRIDTOOLS_STATIC_ASSERT(is_dimension_extension_traits< Storage >::value, "Library internal error");
        typedef typename get_width<
            typename compute_storage_list_width< next_storage_t, Id + 1, IdMax >::next_storage_t >::type type;
        static const uint_t value = type::value;
    };

    // recursion anchor
    template < typename Storage, uint_t IdMax >
    struct compute_storage_list_width< Storage, IdMax, IdMax > {
        GRIDTOOLS_STATIC_ASSERT(IdMax >= 0, "Library internal error");
        GRIDTOOLS_STATIC_ASSERT(is_dimension_extension_traits< Storage >::value, "Library internal error");
        typedef Storage next_storage_t;
        typedef typename get_width< Storage >::type type;
        static const uint_t value = type::value;
    };
    namespace impl_ {
        /**@brief syntactic sugar*/
        template < typename Storage, uint_t Id >
        using offset_t = compute_storage_offset< typename Storage::traits, Id, Storage::traits::n_dimensions - 1 >;

        /**@brief syntactic sugar*/
        template < typename Storage, uint_t Id >
        using width_t = compute_storage_list_width< typename Storage::traits, Id, Storage::traits::n_dimensions - 1 >;
    }

    template < typename T >
    struct get_traits {
        typedef typename T::traits type;
    };

    template < typename First, typename... StorageExtended >
    struct is_data_field< data_field< First, StorageExtended... > > : public boost::mpl::true_ {};

    template < typename T >
    struct storage;

    template < typename First, typename... StorageExtended >
    struct is_data_field< storage< data_field< First, StorageExtended... > > > : public boost::mpl::true_ {};
} // namespace gridtools
