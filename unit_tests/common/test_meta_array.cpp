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

#include "boost/mpl/quote.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/type_traits/is_integral.hpp"
#include "common/defs.hpp"
#include "common/meta_array.hpp"
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
template < typename T >
struct is_integer : boost::mpl::false_ {};
template <>
struct is_integer< int > : boost::mpl::true_ {};

TEST(meta_array, test_meta_array_element) {

    typedef meta_array< boost::mpl::vector4< int, int, int, int >, boost::mpl::quote1< boost::is_integral > >
        meta_array_t;

    ASSERT_TRUE((boost::mpl::equal< meta_array_t::elements, boost::mpl::vector4< int, int, int, int > >::value));

    GRIDTOOLS_STATIC_ASSERT((is_meta_array_of< meta_array_t, boost::is_integral >::value), "Error");
}
