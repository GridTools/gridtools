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
#include "../../common/gt_assert.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/array.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../../common/offset_tuple.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
#include "../iterate_domain_aux.hpp"

#ifdef CXX11_ENABLED
#include "meta_storage_cache.hpp"
#include "cache_storage_metafunctions.hpp"
#endif

namespace gridtools {
    template < typename T, typename U >
    struct get_storage_accessor;

    template < typename BlockSize, typename Extent, typename StorageWrapper >
    struct cache_storage;

    namespace impl_ {

        /** helper function (base case) computing sum(offset*stride ...)*/
        template < unsigned From = 0, unsigned To = 0, typename StorageInfo, typename Accessor >
        GT_FUNCTION constexpr typename boost::enable_if_c< (From == To), int_t >::type get_offset(Accessor acc) {
            return 0;
        }

        /** helper function (step case) computing sum(offset*stride ...)*/
        template < unsigned From = 0, unsigned To = 0, typename StorageInfo, typename Accessor >
        GT_FUNCTION constexpr typename boost::enable_if_c< (From < To), int_t >::type get_offset(Accessor acc) {
            return StorageInfo::template stride< From >() * acc.template get< Accessor::n_dimensions - 1 - From >() +
                   get_offset< From + 1, To, StorageInfo, Accessor >(acc);
        }
    }

    /**
     * @struct cache_storage
     * simple storage class for storing caches. Current version is multidimensional, but allows the user only to cache
     * an entire dimension.
     * Which dimensions to cache is decided by the extents. if the extent is 0,0 the dimension is not cached (and CANNOT
     * BE ACCESSED with an offset other than 0).
     * In a cached data field we suppose that all the snapshots get cached (adding an extra dimension to the
     * meta_storage_cache)
     * in future version we need to support K and IJK storages. Data is allocated on the stack.
     * The size of the storage is determined by the block size and the extension to this block sizes required for
     *  halo regions (determined by a extent type)
     * @tparam Value value type being stored
     * @tparam BlockSize physical domain block size
     * @tparam Extend extent
     * @tparam NColors number of colors of the location type of the storage
     * @tparam Storage type of the storage
     */
    template < uint_t... Tiles, short_t... ExtentBounds, typename StorageWrapper >
    struct cache_storage< block_size< Tiles... >, extent< ExtentBounds... >, StorageWrapper > {

      public:
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::first minus_t;
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::second plus_t;
        typedef variadic_to_vector< static_int< Tiles >... > tiles_t;

        typedef typename StorageWrapper::data_t value_type;

        typedef typename boost::is_same< enumtype::default_location_type,
            typename StorageWrapper::arg_t::location_t >::type is_default_location_t;
        typedef typename _impl::generate_layout_map< typename make_gt_integer_sequence< uint_t,
            sizeof...(Tiles) + (!is_default_location_t::value) >::type >::type layout_t;

        GT_FUNCTION
        explicit constexpr cache_storage() {}

        typedef typename _impl::compute_meta_storage< layout_t, plus_t, minus_t, tiles_t, StorageWrapper >::type meta_t;

        GT_FUNCTION
        static constexpr uint_t size() { return meta_t::size(); }

        template < uint_t Color, typename Accessor >
        GT_FUNCTION value_type &RESTRICT at(array< int, 2 > const &thread_pos, Accessor const &accessor_) {

            using accessor_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< accessor_t >::value), GT_INTERNAL_ERROR_MSG("Error type is not accessor tuple"));

            typedef typename boost::mpl::at_c< typename minus_t::type, 0 >::type iminus;
            typedef typename boost::mpl::at_c< typename minus_t::type, 1 >::type jminus;

            typedef static_int< meta_t::template stride< 0 >() > check_constexpr_1;
            typedef static_int< meta_t::template stride< 1 >() > check_constexpr_2;

            // manually aligning the storage
            const uint_t extra_ =
                (thread_pos[0] - iminus::value) * meta_t::template stride< 0 >() +
                (thread_pos[1] - jminus::value) * meta_t::template stride< 1 + (!is_default_location_t::value) >() +
                (!is_default_location_t::value) * Color * meta_t::template stride< 1 >() +
                size() * get_datafield_offset< typename StorageWrapper::storage_t >::get(accessor_) +
                impl_::get_offset< 0, meta_t::layout_t::masked_length, meta_t >(accessor_);
            assert((extra_) >= 0);
            assert((extra_) < (size() * StorageWrapper::storage_size));
            return m_values[extra_];
        }

      private:
        value_type m_values[size() * StorageWrapper::storage_size];
    };

} // namespace gridtools
