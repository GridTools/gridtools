/*
 * cache_storage.hpp
 *
 *  Created on: Jul 20, 2015
 *      Author: cosuna
 */

#pragma once
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
    typedef typename Range::iminus minusi;
    typedef typename Range::jminus minusj;
    typedef typename Range::iplus plusi;
    typedef typename Range::jplus plusj;

        /**
           constructor of the temporary storage.

           \param initial_offset_i
           \param initial_offset_j
           \param dim3
           \param \optional n_i_threads (Default 1)
           \param \optional n_j_threasd (Default 1)
           \param \optional init (Default value_type())
           \param \optional s (Default "default_name")
         */
    explicit cache_storage() {}

};

} // namespace gridtools
