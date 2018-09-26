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
#include "backend_select.hpp"
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

struct functor1 {
    typedef accessor<0> in;
    typedef accessor<1> out;
    typedef boost::mpl::vector<in, out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {}
};

typedef backend_t::storage_traits_t::storage_info_t<0, 3> meta_data_t;
typedef backend_t::storage_traits_t::data_store_t<float_type, meta_data_t> storage_t;

typedef arg<0, storage_t> p_in;
typedef arg<1, storage_t> p_out;
typedef tmp_arg<2, storage_t> p_buff;

TEST(mss_metafunctions, extract_mss_caches_and_esfs) {
    meta_data_t meta_(10, 10, 10);
    storage_t in(meta_, 1.0), out(meta_, 1.0);

    typedef decltype(make_stage<functor1>(p_in(), p_buff())) esf1_t;
    typedef decltype(make_stage<functor1>(p_buff(), p_out())) esf2_t;

    typedef decltype(make_multistage // mss_descriptor
        (execute<forward>(),
            define_caches(cache<IJ, cache_io_policy::local>(p_buff(), p_out())),
            esf1_t(), // esf_descriptor
            esf2_t()  // esf_descriptor
            )) mss_t;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<mss_t::esf_sequence_t, boost::mpl::vector2<esf1_t, esf2_t>>::value), "ERROR");

#ifndef __DISABLE_CACHING__
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<mss_t::cache_sequence_t,
                                boost::mpl::vector2<detail::cache_impl<IJ, p_buff, cache_io_policy::local, void>,
                                    detail::cache_impl<IJ, p_out, cache_io_policy::local, void>>>::value),
        "ERROR\nLists do not match");
#else
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::empty<mss_t::cache_sequence_t>::value), "ERROR\nList not empty");
#endif

    ASSERT_TRUE(true);
}
