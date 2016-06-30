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
#pragma once
#include <stencil-composition/stencil-composition.hpp>

template < int Arg1, int Arg2 >
struct pair_ {
    static const constexpr int first = Arg1;
    static const constexpr int second = Arg2;
};

GT_FUNCTION
void test_offset_tuple(bool *result) {
    using namespace gridtools;
    *result = true;
#if defined(NDEBUG) && defined(CXX11_ENABLED) && !defined(__CUDACC__)
    {
        constexpr array< int_t, 4 > pos{2, 5, 8, -6};
        constexpr offset_tuple< 4, 4 > offsets(0, pos);

        GRIDTOOLS_STATIC_ASSERT((static_int< offsets.get< 0 >() >::value == -6), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_int< offsets.get< 1 >() >::value == 8), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_int< offsets.get< 2 >() >::value == 5), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_int< offsets.get< 3 >() >::value == 2), "Error");
    }
#endif
    {
#ifdef CXX11_ENABLED
        array< int_t, 4 > pos{2, 5, 8, -6};
#else
        array< int_t, 4 > pos(2, 5, 8, -6);
#endif
        offset_tuple< 4, 4 > offsets(0, pos);

        *result &= ((offsets.get< 0 >() == -6));
        *result &= ((offsets.get< 1 >() == 8));
        *result &= ((offsets.get< 2 >() == 5));
        *result &= ((offsets.get< 3 >() == 2));
    }

#if !defined(__CUDACC__) && defined(CXX11_ENABLED)

    typedef offset_tuple_mixed< boost::mpl::vector< static_int< 1 >, static_int< 2 > >,
        3,
        pair_< 1, 8 >,
        pair_< 2, 7 > > offset_tuple_mixed_t;

    constexpr offset_tuple_mixed_t offset(11, 12, 13);
    GRIDTOOLS_STATIC_ASSERT((static_uint< offset.template get< 2 >() >::value == 8), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((static_uint< offset.template get< 1 >() >::value == 7), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((static_uint< offset.template get< 0 >() >::value == 13), "ERROR");

    assert(offset.template get< 2 >() == 8);
    assert(offset.template get< 1 >() == 7);
    assert(offset.template get< 0 >() == 13);

#endif
}
