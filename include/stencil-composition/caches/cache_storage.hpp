#pragma once
#include "../../common/gt_assert.hpp"
#include "../../common/array.hpp"
#include "../block_size.hpp"
#include "../extent.hpp"
#include "../offset_tuple.hpp"
#include "../extent.hpp"

namespace gridtools {

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
    template < typename Value, typename BlockSize, typename Extend >
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
            GRIDTOOLS_STATIC_ASSERT((is_offset_tuple< Offset >::value), "Error type is not offset tuple");
            assert(index(thread_pos, offset) < storage_size_t::value);
            assert(index(thread_pos, offset) >= 0);

            return m_values[index(thread_pos, offset)];
        }

      private:
        template < typename Offset >
        GT_FUNCTION int_t index(array< int, 2 > const &thread_pos, Offset const &offset) {
            return (thread_pos[0] + offset.template get< Offset::n_args - 1 >() - iminus::value) * i_stride_t::value +
                   (thread_pos[1] + offset.template get< Offset::n_args - 2 >() - jminus::value) * j_stride_t::value;
        }

        Value m_values[storage_size_t::value];
    };

} // namespace gridtools
