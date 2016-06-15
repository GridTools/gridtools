/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "../../common/gt_assert.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/array.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../offset_tuple.hpp"

#ifdef CUDA8
#include "meta_storage_cache.hpp"
#include "cache_storage_metafunctions.hpp"
#endif

namespace gridtools {
#ifdef CUDA8
    template < typename T, typename U >
    struct get_storage_accessor;

    template < typename Value, typename BlockSize, typename Extent, typename Storage >
    struct cache_storage;

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
     */
    template < typename Value, uint_t... Tiles, short_t... ExtentBounds, typename Storage >
    struct cache_storage< Value, block_size< Tiles... >, extent< ExtentBounds... >, Storage > {

      public:
        typedef typename unzip< std::tuple< static_short< ExtentBounds >... > >::first minus_t;
        typedef typename unzip< std::tuple< static_short< ExtentBounds >... > >::second plus_t;
        typedef std::tuple< static_int< Tiles >... > tiles_t;

        // Storage must be a gridtools::pointer to storage
        GRIDTOOLS_STATIC_ASSERT(is_pointer< Storage >::value, "wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_storage< typename Storage::value_type >::value, "wrong type");
        typedef typename Storage::value_type::basic_type storage_t;

        typedef
            typename _impl::generate_layout_map< typename make_gt_integer_sequence< uint_t, sizeof...(Tiles) + 2 /*FD*/
                >::type >::type layout_t;

        GT_FUNCTION
        explicit constexpr cache_storage() {}

        typedef typename _impl::compute_meta_storage< layout_t, plus_t, minus_t, tiles_t, storage_t >::type meta_t;

        // static constexpr const meta_t m_value=meta_t{};

        GT_FUNCTION
        static constexpr uint_t size() { return meta_t{}.size(); }

        template < typename Accessor >
        GT_FUNCTION Value &RESTRICT at(array< int, 2 > const &thread_pos, Accessor const &accessor_) {
            constexpr const meta_t m_value;

            using accessor_t = typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor_t >::value), "Error type is not accessor tuple");

            using iminus = typename std::tuple_element< 0, minus_t >::type;
            using jminus = typename std::tuple_element< 1, minus_t >::type;

            // const typename alias< accessor< Accessor::index_type::value, Intent, Extent, Size >, dimension< Size - 1 > >::template set< ID >
            //     tmp_(arg);
            assert((m_value.index(accessor_) - iminus::value * m_value.template strides< 0 >() - jminus::value * m_value.template strides< 1 >())< size());
            assert((m_value.index(accessor_) - iminus::value * m_value.template strides< 0 >() - jminus::value * m_value.template strides< 1 >() >= 0));

            // manually aligning the storage
            const uint_t extra_ = (thread_pos[0] - iminus::value) * m_value.template strides< 0 >() +
                                  (thread_pos[1] - jminus::value) * m_value.template strides< 1 >() +
                                  m_value.index(accessor_);
            return m_values[extra_];
        }

      private:
        Value m_values[size()];
    };

#else

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
    template < typename Value, typename BlockSize, typename Extend, typename Storage >
    struct cache_storage {

        GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extend >::value), "Internal Error: wrong type");

        typedef typename BlockSize::i_size_t tile_i;
        typedef typename BlockSize::j_size_t tile_j;
        typedef typename Extend::iminus iminus;
        typedef typename Extend::jminus jminus;
        typedef typename Extend::iplus iplus;
        typedef typename Extend::jplus jplus;

        typedef static_uint< tile_i::value - iminus::value + iplus::value > j_stride_t;
        typedef static_uint< 1 > i_stride_t;
        typedef static_uint< (tile_i::value - iminus::value + iplus::value) *
                             (tile_j::value - jminus::value + jplus::value) > storage_size_t;
        explicit cache_storage() {}

        template < typename Offset >
        GT_FUNCTION Value &RESTRICT at(array< int, 2 > const &thread_pos, Offset const &offset) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_offset_tuple< typename Offset::tuple_t >::value), "Error type is not offset tuple");
            assert(index(thread_pos, offset.offsets()) < storage_size_t::value);
            assert(index(thread_pos, offset.offsets()) >= 0);

            return m_values[index(thread_pos, offset.offsets())];
        }

      private:
        template < typename Offset >
        GT_FUNCTION int_t index(array< int, 2 > const &thread_pos, Offset const &offset) {
            return (thread_pos[0] + offset.template get< Offset::n_args - 1 >() - iminus::value) * i_stride_t::value +
                   (thread_pos[1] + offset.template get< Offset::n_args - 2 >() - jminus::value) * j_stride_t::value;
        }

        Value m_values[storage_size_t::value];
    };
#endif

} // namespace gridtools
