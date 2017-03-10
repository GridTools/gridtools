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
#include "../../common/gt_assert.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/array.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../iteration_policy_fwd.hpp"
#include "../../common/offset_tuple.hpp"

#include "meta_storage_cache.hpp"
#include "cache_storage_metafunctions.hpp"
#include "cache_traits.hpp"

namespace gridtools {
    template < typename T, typename U >
    struct get_storage_accessor;

    template < typename Cache, typename BlockSize, typename Extent, uint_t NColors, typename Storage >
    struct cache_storage;

#ifdef CXX11_ENABLED
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
     * @tparam Cache a cache
     * @tparam Value value type being stored
     * @tparam BlockSize physical domain block size
     * @tparam Extend extent
     * @tparam NColors number of colors of the location type of the storage
     * @tparam Storage type of the storage
     */
    template < typename Cache, uint_t... Tiles, short_t... ExtentBounds, typename Storage, uint_t NColors >
    struct cache_storage< Cache, block_size< Tiles... >, extent< ExtentBounds... >, NColors, Storage > {
        GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), "Internal Error");

      public:
        typedef Cache cache_t;
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::first minus_t;
        typedef typename unzip< variadic_to_vector< static_short< ExtentBounds >... > >::second plus_t;
        typedef variadic_to_vector< static_int< Tiles >... > tiles_t;

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

        // Storage must be a gridtools::pointer to storage
        GRIDTOOLS_STATIC_ASSERT(is_pointer< Storage >::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_storage< typename Storage::value_type >::value, "wrong type");
        typedef typename Storage::value_type::basic_type storage_t;
        typedef typename storage_t::value_type value_type;

        // generate a layout map with the number of dimensions of the tiles + 1(snapshots) + 1 (field dimension)
        typedef typename _impl::generate_layout_map< typename make_gt_integer_sequence< uint_t,
            sizeof...(Tiles) + 2 /*FD*/
// TODO ICO_STORAGE in irregular grids we have one more dim for color
#ifndef STRUCTURED_GRIDS
                +
                1
#endif
            >::type >::type layout_t;

        GT_FUNCTION
        explicit constexpr cache_storage() {}

        typedef
            typename _impl::compute_meta_storage< layout_t, plus_t, minus_t, tiles_t, NColors, storage_t >::type meta_t;

        GT_FUNCTION
        static constexpr uint_t size() { return meta_t{}.size(); }

        template < uint_t Color, typename Accessor >
        GT_FUNCTION value_type &RESTRICT at(array< int, 2 > const &thread_pos, Accessor const &accessor_) {
            constexpr const meta_t s_storage_info;

            using accessor_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor_t >::value), "Error type is not accessor tuple");

#ifdef CUDA8
            typedef static_int< s_storage_info.template strides< 0 >() > check_constexpr_1;
            typedef static_int< s_storage_info.template strides< 1 >() > check_constexpr_2;
#else
            assert((_impl::compute_size< NColors, minus_t, plus_t, tiles_t, storage_t >::value == size()));
#endif

            // manually aligning the storage
            const uint_t extra_ = (thread_pos[0] - iminus_t::value) * s_storage_info.template strides< 0 >() +
// TODO ICO_STORAGE
#ifdef STRUCTURED_GRIDS
                                  (thread_pos[1] - jminus_t::value) * s_storage_info.template strides< 1 >() +
#else
                                  Color * s_storage_info.template strides< 1 >() +
                                  (thread_pos[1] - jminus_t::value) * s_storage_info.template strides< 2 >() +
#endif
                                  s_storage_info.index(accessor_);

            assert((extra_) < size());
            assert((extra_) >= 0);

            return m_values[extra_];
        }

        template < typename Accessor >
        GT_FUNCTION value_type const &RESTRICT check_kcache_access(Accessor const &accessor_,
            typename boost::enable_if_c< is_k_cache< cache_t >::value, int >::type = 0) const {

            constexpr const meta_t s_storage_info;

            using accessor_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor_t >::value), "Error type is not accessor tuple");

#ifdef CUDA8
            typedef static_int< s_storage_info.template strides< 0 >() > check_constexpr_1;
            typedef static_int< s_storage_info.template strides< 1 >() > check_constexpr_2;
#else
            assert((_impl::compute_size< minus_t, plus_t, tiles_t, storage_t >::value == size()));
#endif

            assert(s_storage_info.index(accessor_) - kminus_t::value < size());
            assert(s_storage_info.index(accessor_) - kminus_t::value >= 0);
        }

        template < typename Accessor >
        GT_FUNCTION value_type &RESTRICT at(Accessor const &accessor_) {
            check_kcache_access(accessor_);

            constexpr const meta_t s_storage_info;

            return m_values[s_storage_info.index(accessor_) - kminus_t::value];
        }

        template < typename Accessor >
        GT_FUNCTION value_type const &RESTRICT at(Accessor const &accessor_,
            typename boost::enable_if_c< is_k_cache< cache_t >::value, int >::type = 0) const {
            check_kcache_access(accessor_);

            constexpr const meta_t s_storage_info;

            return m_values[s_storage_info.index(accessor_) - kminus_t::value];
        }

        template < typename IterationPolicy >
        GT_FUNCTION void slide() {
            // TODO do not slide if cache interval out of ExecutionPolicy intervals

            GRIDTOOLS_STATIC_ASSERT((Cache::cache_type_t::value == K), "Error: we can only slide KCaches");
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy< IterationPolicy >::value), "Error");

            constexpr uint_t ksize = kplus_t::value - kminus_t::value + 1;
            constexpr uint_t kbegin = (IterationPolicy::value == enumtype::forward) ? 0 : ksize - 1;
            constexpr uint_t kend = (IterationPolicy::value == enumtype::backward) ? ksize - 2 : 1;
            for (int_t k = kbegin; IterationPolicy::condition(k, kend); IterationPolicy::increment(k)) {
                m_values[k] = (IterationPolicy::value == enumtype::forward) ? m_values[k + 1] : m_values[k - 1];
            }
        }

      private:
#if defined(CUDA8)
        value_type m_values[size()];
#else

        value_type m_values[_impl::compute_size< NColors, minus_t, plus_t, tiles_t, storage_t >::value];
#endif
    };

#else // CXX11_ENABLED

    /**
     * @struct cache_storage
     * simple storage class for storing caches. Current version assumes only 2D (i,j), but it will be extented
     * in future version to support K and IJK storages. Data is allocated on the stack.
     * The size of the storage is determined by the block size and the extension to this block sizes required for
     *  halo regions (determined by a extent type)
     * @tparam Value value type being stored
     * @tparam BlockSize physical domain block size
     * @tparam Extend extent
     */
    template < typename Cache, typename BlockSize, typename Extend, uint_t NColors, typename Storage >
    struct cache_storage {
        GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extend >::value), "Internal Error: wrong type");

        typedef typename BlockSize::i_size_t tile_i;
        typedef typename BlockSize::j_size_t tile_j;
        typedef typename Extend::iminus iminus;
        typedef typename Extend::jminus jminus;
        typedef typename Extend::iplus iplus;
        typedef typename Extend::jplus jplus;

        typedef static_uint< 1 > i_stride_t;
        typedef static_uint< tile_i::value - iminus::value + iplus::value > c_stride_t;
        typedef static_uint< c_stride_t::value * NColors > j_stride_t;
        typedef static_uint< j_stride_t::value *(tile_j::value - jminus::value + jplus::value) > storage_size_t;

        typedef typename Storage::value_type::basic_type storage_t;
        typedef typename storage_t::value_type value_type;

        explicit cache_storage() {}

        template < uint_t Color, typename Offset >
        GT_FUNCTION value_type &RESTRICT at(array< int, 2 > const &thread_pos, Offset const &offset) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_offset_tuple< typename Offset::offset_tuple_t >::value), "Error type is not offset tuple");
            assert(index< Color >(thread_pos, offset.offsets()) < storage_size_t::value);
            assert(index< Color >(thread_pos, offset.offsets()) >= 0);

            return m_values[index< Color >(thread_pos, offset.offsets())];
        }

      private:
        template < uint_t Color, typename Offset >
        GT_FUNCTION int_t index(array< int, 2 > const &thread_pos, Offset const &offset) {
            return (thread_pos[0] + offset.template get< Offset::n_args - 1 >() - iminus::value) * i_stride_t::value +
// TODO ICO_STORAGE
#ifdef STRUCTURED_GRIDS
                   (thread_pos[1] + offset.template get< Offset::n_args - 2 >() - jminus::value) * j_stride_t::value;
#else
                   (Color + offset.template get< Offset::n_args - 2 >()) * c_stride_t::value +
                   (thread_pos[1] + offset.template get< Offset::n_args - 3 >() - jminus::value) * j_stride_t::value;
#endif
        }

        value_type m_values[storage_size_t::value];
    };
#endif // CXX11_ENABLED

} // namespace gridtools
