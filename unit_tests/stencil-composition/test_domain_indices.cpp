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
#include "gtest/gtest.h"

#include "stencil-composition/stencil-composition.hpp"
#include <boost/current_function.hpp>

#include "backend_select.hpp"

using namespace gridtools;
using namespace enumtype;

uint_t count;
bool result;

struct print_ {
    print_(void) {}

    template < typename T >
    void operator()(T const &v) const {
        if (T::value != count)
            result = false;
        ++count;
    }
};

struct print_plchld {
    mutable uint_t count;
    mutable bool result;

    print_plchld(void) {}

    template < typename T >
    void operator()(T const &v) const {
        if (T::index_t::value != count) {
            result = false;
        }
        ++count;
    }
};

bool test_domain_indices() {
    typedef gridtools::storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;

    uint_t d1 = 10;
    uint_t d2 = 10;
    uint_t d3 = 10;

    storage_info_t meta_(d1, d2, d3);
    data_store_t in(meta_, -1.);
    data_store_t out(meta_, -7.3);
    data_store_t coeff(meta_, 8.);

    typedef tmp_arg< 2, data_store_t > p_lap;
    typedef tmp_arg< 1, data_store_t > p_flx;
    typedef tmp_arg< 5, data_store_t > p_fly;
    typedef arg< 0, data_store_t > p_coeff;
    typedef arg< 3, data_store_t > p_in;
    typedef arg< 4, data_store_t > p_out;

    result = true;

    typedef boost::mpl::vector< p_lap, p_flx, p_fly, p_coeff, p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain(coeff, in, out);

    count = 0;
    result = true;

    print_plchld pfph;
    count = 0;
    result = true;
    boost::mpl::for_each< aggregator_type< accessor_list >::placeholders_t >(pfph);

    return result;
}

TEST(testdomain, testindices) { EXPECT_EQ(test_domain_indices(), true); }
