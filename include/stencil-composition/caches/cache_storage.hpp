/*
 * cache_storage.hpp
 *
 *  Created on: Jul 20, 2015
 *      Author: cosuna
 */

#pragma once
#include <common/gt_assert.hpp>
#include <stencil-composition/block_size.hpp>
#include <storage/block_storage.hpp>

namespace gridtools {

/**
 * @struct cache_storage
 * simple storage class for storing caches. Current version assumes only 2D (i,j), but it will be extended
 * in future version to support K and IJK storages. Data is allocated on the stack.
 * The size of the storage is determined by the block size and the extension to this block sizes required for
 *  halo regions (determined by a range type)
 * @tparam Value value type being stored
 * @tparam BlockSize physical domain block size
 * @tparam Range range
 */
template <typename Value, typename BlockSize, typename Range>
struct cache_storage : public block_storage<Value, BlockSize, Range>
{

    GRIDTOOLS_STATIC_ASSERT((is_block_size<BlockSize>::value), "Internal Error: wrong type");
    GRIDTOOLS_STATIC_ASSERT((is_range<Range>::value), "Internal Error: wrong type");

    typedef typename BlockSize::i_size_t tile_i;
    typedef typename BlockSize::i_size_t tile_j;
    typedef typename Range::iminus iminus;
    typedef typename Range::jminus jminus;
    typedef typename Range::iplus iplus;
    typedef typename Range::jplus jplus;

    typedef static_uint<tile_i::value - iminus::value + iplus::value> j_stride_t;
    typedef static_uint<1> i_stride_t;
    explicit cache_storage() {}

    template<typename Offset>
    GT_FUNCTION
    Value& RESTRICT at(array<int, 2> const & thread_pos, Offset const & offset)
    {
        GRIDTOOLS_STATIC_ASSERT((is_offset_tuple<Offset>::value), "Error type is not offset tuple");
        // TODO assert not working, problem with PRETTY_FUNCTION
//        assert(true);
        return m_values[(thread_pos[0] + offset.template get<Offset::n_args-1>() - iminus::value) * i_stride_t::value +
                (thread_pos[1] + offset.template get<Offset::n_args-2>() -  jminus::value) * j_stride_t::value];
    }

private:
    Value m_values[(tile_i::value-iminus::value+iplus::value)*(tile_j::value-jminus::value+jminus::value)];
};

} // namespace gridtools
