/*
 * block_storage.h
 *
 *  Created on: Jul 20, 2015
 *      Author: cosuna
 */

#pragma once
#include <common/defs.hpp>

namespace gridtools{

template<typename Value, typename BlockSize, typename Range>
struct block_storage
{
    typedef typename BlockSize::i_size_t itile_t;
    typedef typename BlockSize::i_size_t jtile_t;
    typedef typename Range::iminus iminus_t;
    typedef typename Range::jminus jminus_t;
    typedef typename Range::iplus iplus_t;
    typedef typename Range::jplus jplus_t;

    typedef static_int< (itile_t::value-iminus_t::value+iplus_t::value)*(jtile_t::value-jminus_t::value+jplus_t::value) > size_t;
private:
    Value data[size_t::value];
};

} //namespace gridtools
