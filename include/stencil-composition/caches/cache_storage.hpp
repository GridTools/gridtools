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

    template<typename T> struct printy{BOOST_MPL_ASSERT_MSG((false), IIIIIIIII, (T));};
    template<typename Offset>
    GT_FUNCTION
    Value& at(const array<int, 2> thread_pos, Offset const & offset)
    {
        GRIDTOOLS_STATIC_ASSERT((is_offset_tuple<Offset>::value), "Error type is not offset tuple");
        // TODO assert not working, problem with PRETTY_FUNCTION
//        assert(true);
        return m_values[(thread_pos[0] + offset.template get<0>() - iminus::value) * i_stride_t::value +
                (thread_pos[0] + offset.template get<0>() + jminus::value) * i_stride_t::value];
    }

private:
    Value m_values[(tile_i::value-iminus::value+iplus::value)*(tile_j::value-jminus::value+jminus::value)];
};

} // namespace gridtools
