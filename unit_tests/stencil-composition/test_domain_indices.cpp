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
#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include "gtest/gtest.h"

#include "stencil-composition/stencil-composition.hpp"
#include <boost/current_function.hpp>

using namespace gridtools;
using namespace enumtype;

uint_t count;
bool result;

struct print_ {
    print_(void)
    {}

    template <typename T>
    void operator()(T const& v) const {
        if (T::value != count)
            result = false;
        ++count;
    }
};

struct print_plchld {
    mutable uint_t count;
    mutable bool result;

    print_plchld(void)
    {}

    template <typename T>
    void operator()(T const& v) const {
        if (T::index_type::value != count) {
            result = false;
        }
        ++count;
    }
};

bool test_domain_indices() {

    typedef backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_type< float_type,
        backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_info< 0, layout_map< 0, 1, 2 > > >::type
        storage_type;
    typedef backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::temporary_storage_type< float_type,
        backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_info< 0, layout_map< 0, 1, 2 > > >::type
        tmp_storage_type;

    uint_t d1 = 10;
    uint_t d2 = 10;
    uint_t d3 = 10;

    backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >::storage_info< 0, layout_map< 0, 1, 2 > > meta_(d1, d2, d3);
    storage_type in(meta_,-1., "in");
    storage_type out(meta_,-7.3, "out");
    storage_type coeff(meta_,8., "coeff");

    typedef arg<2, tmp_storage_type > p_lap;
    typedef arg<1, tmp_storage_type > p_flx;
    typedef arg<5, tmp_storage_type > p_fly;
    typedef arg<0, storage_type > p_coeff;
    typedef arg<3, storage_type > p_in;
    typedef arg<4, storage_type > p_out;

    result = true;

    typedef boost::mpl::vector<p_lap, p_flx, p_fly, p_coeff, p_in, p_out> accessor_list;

    aggregator_type<accessor_list> domain
       (boost::fusion::make_vector(&out, &in, &coeff /*,&fly, &flx*/));

    count = 0;
    result = true;

    print_plchld pfph;
    count = 0;
    result = true;
    boost::mpl::for_each<aggregator_type<accessor_list>::placeholders>(pfph);


    return result;
}

TEST(testdomain, testindices) {
    EXPECT_EQ(test_domain_indices(), true);
}

