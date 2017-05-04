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

#include <iostream>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>

#include "storage/wrap_pointer.hpp"
#ifdef _USE_GPU_
#include "storage/hybrid_pointer.hpp"
#endif
#include "common/pointer_metafunctions.hpp"

using namespace gridtools;

TEST(wrap_hybrid_pointer_metafunctions, test_pointer) {
    typedef wrap_pointer< double, true > wpt;
    typedef wrap_pointer< double, false > wpf;

    GRIDTOOLS_STATIC_ASSERT(!is_wrap_pointer< double >::value, "is_wrap_pointer<double> failed");
    GRIDTOOLS_STATIC_ASSERT(
        is_wrap_pointer< wrap_pointer< double > >::value, "is_wrap_pointer<wrap_pointer<double> > failed");
    GRIDTOOLS_STATIC_ASSERT(is_wrap_pointer< wpt >::value, "is_wrap_pointer<wrap_pointer<double, true> > failed");
    GRIDTOOLS_STATIC_ASSERT(is_wrap_pointer< wpf >::value, "is_wrap_pointer<wrap_pointer<double, false> > failed");

    GRIDTOOLS_STATIC_ASSERT(!is_hybrid_pointer< double >::value, "is_hybrid_pointer<double> failed");
    GRIDTOOLS_STATIC_ASSERT(
        !is_hybrid_pointer< wrap_pointer< double > >::value, "is_hybrid_pointer<wrap_pointer<double> > failed");
    GRIDTOOLS_STATIC_ASSERT(!is_hybrid_pointer< wpt >::value, "is_hybrid_pointer<wrap_pointer<double, true> > failed");
    GRIDTOOLS_STATIC_ASSERT(!is_hybrid_pointer< wpf >::value, "is_hybrid_pointer<wrap_pointer<double, false> > failed");

#ifdef _USE_GPU_
    typedef hybrid_pointer< double, true > hpt;
    typedef hybrid_pointer< double, false > hpf;

    GRIDTOOLS_STATIC_ASSERT(
        !is_wrap_pointer< hybrid_pointer< double > >::value, "is_wrap_pointer<wrap_pointer<double> > failed");
    GRIDTOOLS_STATIC_ASSERT(!is_wrap_pointer< hpt >::value, "is_wrap_pointer<wrap_pointer<double, true> > failed");
    GRIDTOOLS_STATIC_ASSERT(!is_wrap_pointer< hpf >::value, "is_wrap_pointer<wrap_pointer<double, false> > failed");

    GRIDTOOLS_STATIC_ASSERT(
        is_hybrid_pointer< hybrid_pointer< double > >::value, "is_hybrid_pointer<wrap_pointer<double> > failed");
    GRIDTOOLS_STATIC_ASSERT(is_hybrid_pointer< hpt >::value, "is_hybrid_pointer<wrap_pointer<double, true> > failed");
    GRIDTOOLS_STATIC_ASSERT(is_hybrid_pointer< hpf >::value, "is_hybrid_pointer<wrap_pointer<double, false> > failed");

    typedef hybrid_pointer< int > h_pointer_type;
    typedef wrap_pointer< int > w_pointer_type;

    typedef typename boost::mpl::if_< is_wrap_pointer< h_pointer_type >,
        wpf,
        typename boost::mpl::if_< is_hybrid_pointer< h_pointer_type >, hpf, boost::mpl::void_ >::type >::type
        h_storage_ptr_t;

    typedef typename boost::mpl::if_< is_wrap_pointer< w_pointer_type >,
        wpf,
        typename boost::mpl::if_< is_hybrid_pointer< w_pointer_type >, hpf, boost::mpl::void_ >::type >::type
        w_storage_ptr_t;

    GRIDTOOLS_STATIC_ASSERT((boost::is_same< h_storage_ptr_t, hpf >::value), "boost::if test case failed");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< w_storage_ptr_t, wpf >::value), "boost::if test case failed");
#endif
}
