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

    /**@brief local class instead of using the inline (cond)?a:b syntax, because in the latter both branches get
     * compiled (generating sometimes a compile-time overflow) */
    template < bool condition, typename LocalD, typename Accessor >
    struct current_storage;

    template < typename LocalD, typename Accessor >
    struct current_storage< true, LocalD, Accessor > {
        static const uint_t value = 0;
    };

    template < typename LocalD, typename Accessor >
    struct current_storage< false, LocalD, Accessor > {
        static const uint_t value =
            (total_storages< typename LocalD::local_storage_type, Accessor::index_type::value >::value);
    };

    template < bool cond, typename Accessor, typename LocalDomain >
    struct get_data_field_index;

    /** rectangular data field */
    template < typename Accessor, typename LocalDomain >
    struct get_data_field_index< true, Accessor, LocalDomain > {

        typedef typename Accessor::index_type index_t;
        typedef typename LocalDomain::template get_storage< index_t >::type::value_type storage_type;
        typedef typename storage_type::storage_info_type metadata_t;

        static constexpr uint_t apply(Accessor const &accessor_) {

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dim >= storage_type::storage_info_type::space_dimensions + 1,
                "dimensionality error in the storage accessor. increase the number of dimensions when "
                "defining the accessor.");

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dim <= storage_type::storage_info_type::space_dimensions + 2,
                "dimensionality error in the storage accessor. decrease the number of dimensions when "
                "defining the accessor.");

            GRIDTOOLS_STATIC_ASSERT((storage_type::traits::n_fields % storage_type::traits::n_width == 0),
                "You specified a non-rectangular field: if you need to use a non-rectangular field the constexpr "
                "version of the accessors have to be used (so that the current position in the field is computed at "
                "compile time). This is achieved by using, e.g., instead of \n\n eval(field(dimension<5>(2))); \n\n "
                "the following expression: \n\n typedef alias<field, dimension<5> >::set<2> z_field; \n "
                "eval(z_field()); \n");

            // dimension/snapshot offsets must be non negative
            //             GTASSERT(accessor_.template get< 0 >() >= 0);

            // if number of accessor dimensions is equal to the number of space dimensions+1
            // we have a storage list of snapshots, i.e. one dimensional array of storages:
            // just return the last offset (get<0>()) + the index of the current data_field in the array of storages
            // contained in the iterate_domain
            //
            // if number of accessor dimensions is larger than the space_dimensions (i.e. normal dimensions + color)
            // get the last-1 offset, sum it to the last offset times the #snapshots (storage_type::traits::n_width)
            // sum the index of the current data_field in the array of storages contained in the iterate_domain

            return (Accessor::type::n_dim <= metadata_t::space_dimensions + 1
                           ?                               // static if
                           accessor_.template get< 0 >()   // offset for the current snapshot
                           : accessor_.template get< 1 >() // offset for the current snapshot
                                 // limitation to "rectangular" vector fields for non-static fields dimensions
                                 +
                                 accessor_.template get< 0 >() // select the dimension
                                     *
                                     storage_type::traits::n_width // stride of the current dimension inside
                                                                   // the vector
                                                                   // of
                       // storages
                       ) + //+ the offset of the other extra dimension
                   current_storage< (Accessor::index_type::value == 0), LocalDomain, Accessor >::value;
        }
    };

    /** non rectangular data field */
    template < typename Accessor, typename LocalDomain >
    struct get_data_field_index< false, Accessor, LocalDomain > {

        static constexpr uint_t apply(Accessor const &accessor_) {

            typedef typename Accessor::index_type index_t;
            typedef typename LocalDomain::template get_storage< index_t >::type::value_type storage_t;
            typedef typename storage_t::storage_info_type metadata_t;

            GRIDTOOLS_STATIC_ASSERT(Accessor::template get_constexpr< 0 >() >= 0,
                "offset specified for the dimension corresponding to the number of field components/snapshots must be "
                "non "
                "negative");

            // "offset specified for the dimension corresponding to the number of snapshots must be non negative"
            GRIDTOOLS_STATIC_ASSERT(((Accessor::n_dim <= metadata_t::space_dimensions + 1) ||
                                        (Accessor::template get_constexpr< 1 >() >= 0)),
                "offset specified for the dimension corresponding to the number of snapshots must be non negative");
            GRIDTOOLS_STATIC_ASSERT(
                (storage_t::traits::n_width > 0), "did you define a field dimension with 0 snapshots??");
            // "field dimension access out of bounds"
            GRIDTOOLS_STATIC_ASSERT(((Accessor::template get_constexpr< 0 >() < storage_t::traits::n_dimensions) ||
                                        Accessor::n_dim <= metadata_t::space_dimensions + 1),
                "field dimension access out of bounds");

            // snapshot access out of bounds
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::template get_constexpr< 1 >() <
                    _impl::access< storage_t::n_width - (Accessor::template get_constexpr< 0 >()) - 1,
                        typename storage_t::traits >::type::n_width),
                "snapshot access out of bounds");

            return (Accessor::type::n_dim <= metadata_t::space_dimensions + 1
                           ?                                         // static if
                           Accessor::template get_constexpr< 0 >()   // offset for the current snapshot
                           : Accessor::template get_constexpr< 1 >() // offset for the current snapshot
                                 // hypotheses : storage offsets are known at compile-time
                                 +
                                 compute_storage_offset< typename storage_t::traits,
                                     Accessor::template get_constexpr< 0 >(),
                                     storage_t::traits::n_dimensions -
                                         1 >::value // stride of the current dimension inside the vector of storages
                       ) +                          //+ the offset of the other extra dimension
                   current_storage< (Accessor::index_type::value == 0), LocalDomain, Accessor >::value;
        }
    };
} // namespace gridtools
