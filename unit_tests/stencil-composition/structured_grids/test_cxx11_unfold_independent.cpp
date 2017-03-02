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
#include <stencil-composition/stencil-composition.hpp>

struct functor {
    using a0 = gridtools::accessor< 0, gridtools::enumtype::inout >;
    using a1 = gridtools::accessor< 1, gridtools::enumtype::inout >;

    typedef boost::mpl::vector< a0, a1 > arg_list;
};

struct fake_storage_type {
    using value_type = int;
    using iterator = int *;
};

TEST(unfold_independent, test) {

    using namespace gridtools;

    typedef arg< 0, fake_storage_type > p0;
    typedef arg< 1, fake_storage_type > p1;

    using esf_type = decltype(make_stage< functor >(p0(), p1()));

    using mss_type = decltype(
        make_multistage(enumtype::execute< enumtype::forward >(),
            make_stage< functor >(p0(), p1()),
            make_stage< functor >(p0(), p1()),
            make_stage< functor >(p0(), p1()),
            make_independent(make_stage< functor >(p0(), p1()),
                            make_stage< functor >(p0(), p1()),
                            make_independent(make_stage< functor >(p0(), p1()), make_stage< functor >(p0(), p1())))));

    using sequence = unwrap_independent< mss_type::esf_sequence_t >::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< sequence >::type::value == 7), "");

    GRIDTOOLS_STATIC_ASSERT((is_sequence_of< sequence, is_esf_descriptor >::value), "");
}
