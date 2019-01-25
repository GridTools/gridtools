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

#include <gridtools/stencil-composition/caches/cache_metafunctions.hpp>
#include <gridtools/stencil-composition/caches/extract_extent_caches.hpp>

#include <boost/fusion/include/pair.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace enumtype;

typedef storage_traits<target::x86>::storage_info_t<0, 2> storage_info_ij_t;
typedef storage_traits<target::x86>::data_store_t<float_type, storage_info_ij_t> storage_type;

typedef arg<0, storage_type> p_in;
typedef arg<2, storage_type> p_out;
typedef arg<1, storage_type> p_buff;
typedef arg<3, storage_type> p_notin;

struct functor2 {
    typedef accessor<0, enumtype::in, extent<0, 0, 0, 0, -1, 0>> in;
    typedef accessor<1, enumtype::inout, extent<0, 0, 0, 0, 0, 1>> out;
    typedef boost::mpl::vector<in, out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval);
};

typedef detail::cache_impl<IJ, p_in, cache_io_policy::fill> cache1_t;
typedef detail::cache_impl<IJ, p_buff, cache_io_policy::fill> cache2_t;
typedef detail::cache_impl<K, p_out, cache_io_policy::local> cache3_t;
typedef detail::cache_impl<K, p_notin, cache_io_policy::local> cache4_t;
typedef std::tuple<cache1_t, cache2_t, cache3_t, cache4_t> caches_t;

TEST(cache_metafunctions, get_ij_cache_storage_tuple) {
    using testee_t = get_ij_cache_storage_tuple<caches_t, extent<-2, 2, -3, 2>, 32, 4>::type;

    using expected_t = std::tuple<boost::fusion::pair<p_in, ij_cache_storage<double, 36, 9, 2, 3>>,
        boost::fusion::pair<p_buff, ij_cache_storage<double, 36, 9, 2, 3>>>;

    static_assert(std::is_same<testee_t, expected_t>::value, "");
}

using esf1k_t = decltype(make_stage<functor2>(p_in(), p_notin()));
using esf2k_t = decltype(make_stage<functor2>(p_notin(), p_out()));

using esfk_sequence_t = meta::list<esf1k_t, esf2k_t>;

static_assert(
    std::is_same<GT_META_CALL(extract_k_extent_for_cache, (p_out, esfk_sequence_t)), extent<0, 0, 0, 0, 0, 1>>(), "");

static_assert(
    std::is_same<GT_META_CALL(extract_k_extent_for_cache, (p_notin, esfk_sequence_t)), extent<0, 0, 0, 0, -1, 1>>(),
    "");

TEST(cache_metafunctions, get_k_cache_storage_tuple) {

    using testee_t = typename get_k_cache_storage_tuple<caches_t, esfk_sequence_t>::type;

    using expected_t = std::tuple<boost::fusion::pair<p_out, k_cache_storage<double, 0, 1>>,
        boost::fusion::pair<p_notin, k_cache_storage<double, -1, 1>>>;

    static_assert(std::is_same<testee_t, expected_t>::value, "");
}
