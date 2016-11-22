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

#include "gtest/gtest.h"
#include "stencil-composition/esf_metafunctions.hpp"

using namespace gridtools;

template < int >
struct myt {};

TEST(copy_into_set, all_elements_unique) {
    typedef boost::mpl::vector< myt< 0 >, myt< 1 > > my_vec1;
    typedef boost::mpl::vector< myt< 2 >, myt< 3 > > my_vec2;

    typedef boost::mpl::set< my_vec1, my_vec2 > set_of_vecs;

    typedef typename boost::mpl::fold< set_of_vecs,
        boost::mpl::set0<>,
        copy_into_set< boost::mpl::_2, boost::mpl::_1 > >::type result;

    ASSERT_EQ(4, boost::mpl::size< result >::type::value);
    ASSERT_TRUE((boost::mpl::contains< result, myt< 0 > >::type::value));
    ASSERT_TRUE((boost::mpl::contains< result, myt< 1 > >::type::value));
    ASSERT_TRUE((boost::mpl::contains< result, myt< 2 > >::type::value));
    ASSERT_TRUE((boost::mpl::contains< result, myt< 3 > >::type::value));
}

TEST(copy_into_set, repeating_element) {
    typedef boost::mpl::vector< myt< 0 >, myt< 1 > > my_vec1;
    typedef boost::mpl::vector< myt< 2 >, myt< 0 > > my_vec2;

    typedef boost::mpl::set< my_vec1, my_vec2 > set_of_vecs;

    typedef typename boost::mpl::fold< set_of_vecs,
        boost::mpl::set0<>,
        copy_into_set< boost::mpl::_2, boost::mpl::_1 > >::type result;

    ASSERT_EQ(3, boost::mpl::size< result >::type::value);
    ASSERT_TRUE((boost::mpl::contains< result, myt< 0 > >::type::value));
    ASSERT_TRUE((boost::mpl::contains< result, myt< 1 > >::type::value));
    ASSERT_TRUE((boost::mpl::contains< result, myt< 2 > >::type::value));
}
