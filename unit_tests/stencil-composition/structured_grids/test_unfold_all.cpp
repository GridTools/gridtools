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
#include <gridtools/stencil-composition/conditionals/if_.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>

template <gridtools::uint_t Id>
struct functor {

    typedef gridtools::accessor<0, gridtools::enumtype::inout> a0;
    typedef gridtools::accessor<1, gridtools::enumtype::in> a1;
    typedef boost::mpl::vector2<a0, a1> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {}
};

bool predicate() { return false; }

TEST(unfold_all, test) {

    using namespace gridtools;

    auto grid = make_grid(2, 2, 3);

    typedef backend_t::storage_traits_t::storage_info_t<0, 3> meta_data_t;
    typedef backend_t::storage_traits_t::data_store_t<float_type, meta_data_t> storage_t;

    typedef arg<0, storage_t> p0;
    typedef arg<1, storage_t> p1;

    auto mss1 = make_multistage(enumtype::execute<enumtype::forward>(),
        make_stage<functor<0>>(p0(), p1()),
        make_stage<functor<1>>(p0(), p1()),
        make_stage<functor<2>>(p0(), p1()),
        make_independent(make_stage<functor<3>>(p0(), p1()),
            make_stage<functor<4>>(p0(), p1()),
            make_independent(make_stage<functor<5>>(p0(), p1()), make_stage<functor<6>>(p0(), p1()))));

    auto mss2 = make_multistage(enumtype::execute<enumtype::forward>(),
        make_stage<functor<7>>(p0(), p1()),
        make_stage<functor<8>>(p0(), p1()),
        make_stage<functor<9>>(p0(), p1()),
        make_independent(make_stage<functor<10>>(p0(), p1()),
            make_stage<functor<11>>(p0(), p1()),
            make_independent(make_stage<functor<12>>(p0(), p1()), make_stage<functor<13>>(p0(), p1()))));

    make_computation<backend_t>(grid, if_(predicate, mss1, mss2));
}
