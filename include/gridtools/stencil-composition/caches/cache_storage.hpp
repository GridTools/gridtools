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
#include <boost/utility/enable_if.hpp>
#include "../../common/gt_assert.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/array.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
#include "../iterate_domain_aux.hpp"
#include "../offset_computation.hpp"
#include "cache_traits.hpp"
#include "../iteration_policy_fwd.hpp"

#include "meta_storage_cache.hpp"
#include "cache_storage_metafunctions.hpp"

namespace gridtools {

    namespace _impl {
        template < uint_t TileI, uint_t TileJ, uint_t... Tiles >
        struct check_cache_tile_sizes {
            GRIDTOOLS_STATIC_ASSERT((TileI > 0 && TileJ > 0), GT_INTERNAL_ERROR);
            static constexpr bool value = (accumulate(multiplies(), Tiles...) == 1);
        };
    }

    /**
     * @struct cache_storage
     * simple storage class for storing caches. Current version is multidimensional, but allows the user only to cache
     * an entire dimension.
     * Which dimensions to cache is decided by the extents.
     * In a cached data field we suppose that all the snapshots get cached (adding an extra dimension to the
     * meta_storage_cache)
     * The size of the storage is determined by the block size and the extension to this block sizes required for
     *  halo regions (determined by a extent type)
     * @tparam Cache a cache_impl type of the cache for which this class provides storage functionality
     * @tparam BlockSize physical block size (in IJ dims) that determines the size of the cache storage in the
     * scratchpad
     * @tparam Extent extent at which the cache is used (used also to determine the size of storage)
     * @tparam StorageWrapper storage wrapper containing the storage of the arg being cached
     */
    template < typename Cache, typename BlockSize, typename Extent, typename StorageWrapper >
    struct cache_storage;

    template < typename Cache, uint_t... Tiles, short_t... ExtentBounds, typename StorageWrapper >
    struct cache_storage< Cache, block_size< Tiles... >, extent< ExtentBounds... >, StorageWrapper > {
        GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((_impl::check_cache_tile_sizes< Tiles... >::value), GT_INTERNAL_ERROR);

      public:
        using cache_t = Cache;
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::first minus_t;
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::second plus_t;
        typedef variadic_to_vector< static_int< Tiles >... > tiles_t;

        static constexpr int tiles_block = accumulate(multiplies(), Tiles...);

        GRIDTOOLS_STATIC_ASSERT(((tiles_block == 1) || !is_k_cache< cache_t >::value), GT_INTERNAL_ERROR);

        typedef typename StorageWrapper::data_t value_type;

// TODO ICO_STORAGE in irregular grids we have one more dim for color
#ifndef STRUCTURED_GRIDS
        static constexpr int extra_dims = 1;
#else
        static constexpr int extra_dims = 0;
#endif

        typedef typename _impl::generate_layout_map<
            typename make_gt_integer_sequence< uint_t, sizeof...(Tiles) + (extra_dims) >::type >::type layout_t;

        using iminus_t = typename boost::mpl::at_c< typename minus_t::type, 0 >::type;
        using jminus_t = typename boost::mpl::at_c< typename minus_t::type, 1 >::type;
        using kminus_t = typename boost::mpl::at_c< typename minus_t::type, 2 >::type;
        using iplus_t = typename boost::mpl::at_c< typename plus_t::type, 0 >::type;
        using jplus_t = typename boost::mpl::at_c< typename plus_t::type, 1 >::type;
        using kplus_t = typename boost::mpl::at_c< typename plus_t::type, 2 >::type;

        GRIDTOOLS_STATIC_ASSERT((Cache::cache_type_t::value != K) || (iminus_t::value == 0 && jminus_t::value == 0 &&
                                                                         iplus_t::value == 0 && jplus_t::value == 0),
            "KCaches can not be use with a non null extent in the horizontal dimensions");

        GRIDTOOLS_STATIC_ASSERT((Cache::cache_type_t::value != IJ) || (kminus_t::value == 0 && kplus_t::value == 0),
            "Only KCaches can be accessed with a non null extent in K");

        template < typename Accessor >
        struct is_acc_k_cache : is_k_cache< cache_t > {};

        GT_FUNCTION
        explicit constexpr cache_storage() {}

        typedef typename _impl::compute_meta_storage< layout_t, plus_t, minus_t, tiles_t, StorageWrapper >::type meta_t;

        GT_FUNCTION
        static constexpr uint_t padded_total_length() { return meta_t::padded_total_length(); }

        template < uint_t Color, typename Accessor >
        GT_FUNCTION value_type &RESTRICT at(array< int, 2 > const &thread_pos, Accessor const &accessor_) {

            using accessor_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< accessor_t >::value), GT_INTERNAL_ERROR_MSG("Error type is not accessor tuple"));

            typedef static_int< meta_t::template stride< 0 >() > check_constexpr_1;
            typedef static_int< meta_t::template stride< 1 >() > check_constexpr_2;

            // manually aligning the storage
            const uint_t extra_ =
                (thread_pos[0] - iminus_t::value) * meta_t::template stride< 0 >() +
                (thread_pos[1] - jminus_t::value) * meta_t::template stride< 1 + (extra_dims) >() +
                (extra_dims)*Color * meta_t::template stride< 1 >() +
                padded_total_length() * get_datafield_offset< typename StorageWrapper::data_store_t >::get(accessor_) +
                compute_offset_cache< meta_t >(accessor_);
            assert((extra_) < (padded_total_length() * StorageWrapper::num_of_storages));
            return m_values[extra_];
        }

        /**
         * @brief retrieve value in a cache given an accessor for a k cache
         * @param accessor_ the accessor that contains the offsets being accessed
         */
        template < typename Accessor >
        GT_FUNCTION value_type &RESTRICT at(Accessor const &accessor_,
            typename boost::enable_if_c< is_acc_k_cache< Accessor >::value, int >::type = 0) {
            check_kcache_access(accessor_);

            const int_t index_ =
                (int_t)padded_total_length() *
                    (int_t)get_datafield_offset< typename StorageWrapper::data_store_t >::get(accessor_) +
                compute_offset_cache< meta_t >(accessor_) - kminus_t::value;
            assert(index_ >= 0);
            assert(index_ < (padded_total_length() * StorageWrapper::num_of_storages));

            return m_values[index_];
        }

        /**
         * @brief retrieve value in a cache given an accessor for a k cache
         * @param accessor_ the accessor that contains the offsets being accessed
         */
        template < typename Accessor >
        GT_FUNCTION value_type const &RESTRICT at(Accessor const &accessor_,
            typename boost::enable_if_c< is_acc_k_cache< Accessor >::value, int >::type = 0) const {
            check_kcache_access(accessor_);

            const int_t index_ =
                (int_t)padded_total_length() *
                    (int_t)get_datafield_offset< typename StorageWrapper::data_store_t >::get(accessor_) +
                compute_offset_cache< meta_t >(accessor_) - kminus_t::value;

            assert(index_ >= 0);
            assert(index_ < (padded_total_length() * StorageWrapper::num_of_storages));

            return m_values[index_];
        }

        /**
         * @brief slides the values of the ring buffer
         */
        template < typename IterationPolicy >
        GT_FUNCTION void slide() {
            GRIDTOOLS_STATIC_ASSERT((Cache::cache_type_t::value == K), "Error: we can only slide KCaches");
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "Error");

            constexpr uint_t ksize = kplus_t::value - kminus_t::value + 1;
            if (ksize > 1) {

                constexpr int_t kbegin = (IterationPolicy::value == enumtype::forward) ? 0 : (int_t)ksize - 1;
                constexpr int_t kend = (IterationPolicy::value == enumtype::forward) ? (int_t)ksize - 2 : 1;

                for (int_t k = kbegin; IterationPolicy::condition(k, kend); IterationPolicy::increment(k)) {
                    m_values[k] = (IterationPolicy::value == enumtype::forward) ? m_values[k + 1] : m_values[k - 1];
                }
            }
        }

      private:
        value_type m_values[padded_total_length() * StorageWrapper::num_of_storages];

        template < typename Accessor, std::size_t... Coordinates >
        GT_FUNCTION static void check_kcache_access_in_bounds(
            Accessor const &accessor, gt_index_sequence< Coordinates... >) {
            assert(accumulate(logical_and(),
                       (accessor.template get< Accessor::n_dimensions - 1 - Coordinates >() <=
                                  meta_t::template dim< Coordinates >())...) &&
                   "Out of bounds access in cache");
        }

        template < typename Accessor >
        GT_FUNCTION static void check_kcache_access(
            Accessor const &accessor, typename boost::enable_if_c< is_acc_k_cache< Accessor >::value, int >::type = 0) {

            using accessor_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor_t >::value), "Error type is not accessor tuple");

            typedef static_int< meta_t::template stride< 0 >() > check_constexpr_1;
            typedef static_int< meta_t::template stride< 1 >() > check_constexpr_2;

            check_kcache_access_in_bounds(accessor, make_gt_index_sequence< meta_t::layout_t::masked_length >());
        }
    };

} // namespace gridtools
