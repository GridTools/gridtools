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
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/conditionals/condition_pool.hpp>
#include <stencil-composition/iterate_on_esfs.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

namespace iterate_on_esfs_detail {
    template < int I >
    struct functor {
        static const int thevalue = I;

        typedef in_accessor< 0, extent<>, 3 > in;
        typedef inout_accessor< 1, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

} // namespace iterate_on_esfs_detail

class iterate_on_esfs_class : public testing::Test {
  public:
    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;

    typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;

    typedef arg< 0, data_store_t > p_in;
    typedef arg< 1, data_store_t > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 1;

    storage_info_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< axis > grid;

    data_store_t in;
    data_store_t out;

    aggregator_type< accessor_list > domain;

    iterate_on_esfs_class()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2), grid(di, dj),
          in(meta_, [](int i, int j, int k) { return i + j * 10 + k * 100; }), out(meta_, -5), domain(in, out) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;
    }
};

template < typename StencilOp >
struct is_even {
    using type = boost::mpl::int_< (StencilOp::esf_function::thevalue % 2 == 0) ? 1 : 0 >;
};

template < typename StencilOp >
struct is_odd {
    using type = boost::mpl::int_< (StencilOp::esf_function::thevalue % 2 == 0) ? 0 : 1 >;
};

template < typename A, typename B >
struct sum {
    using type = boost::mpl::int_< A::value + B::value >;
};

TEST_F(iterate_on_esfs_class, basic) {
    auto x = make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 0 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 1 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 2 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 3 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 4 > >(p_in(), p_out())));

    auto e = with_operators<is_even, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    ASSERT_TRUE(e == 3);
    auto o = with_operators<is_odd, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    ASSERT_TRUE(o == 2);
}

TEST_F(iterate_on_esfs_class, two_multistages) {
    auto x = make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 0 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 1 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 2 > >(p_in(), p_out())),
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 3 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 4 > >(p_in(), p_out())));

    auto e = with_operators<is_even, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    ASSERT_TRUE(e == 3);
    auto o = with_operators<is_odd, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    ASSERT_TRUE(o == 2);
}

TEST_F(iterate_on_esfs_class, two_multistages_independent) {
    auto x = make_computation< gridtools::BACKEND >(
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 0 > >(p_in(), p_out()),
            make_independent(gridtools::make_stage< iterate_on_esfs_detail::functor< 1 > >(p_in(), p_out()),
                                       gridtools::make_stage< iterate_on_esfs_detail::functor< 2 > >(p_in(), p_out()))),
        gridtools::make_multistage(execute< forward >(),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 3 > >(p_in(), p_out()),
            gridtools::make_stage< iterate_on_esfs_detail::functor< 4 > >(p_in(), p_out())));

    auto e = with_operators<is_even, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    ASSERT_TRUE(e == 3);
    auto o = with_operators<is_odd, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    ASSERT_TRUE(o == 2);
}

TEST_F(iterate_on_esfs_class, conditionals) {
    auto cond = new_cond([]() { return false; });

    auto x = make_computation< gridtools::BACKEND >(
        domain,
        grid,
        if_(cond,
            gridtools::make_multistage(
                execute< forward >(),
                gridtools::make_stage< iterate_on_esfs_detail::functor< 0 > >(p_in(), p_out()),
                make_independent(gridtools::make_stage< iterate_on_esfs_detail::functor< 1 > >(p_in(), p_out()),
                    gridtools::make_stage< iterate_on_esfs_detail::functor< 2 > >(p_in(), p_out()))),
            gridtools::make_multistage(execute< forward >(),
                gridtools::make_stage< iterate_on_esfs_detail::functor< 3 > >(p_in(), p_out()),
                gridtools::make_stage< iterate_on_esfs_detail::functor< 4 > >(p_in(), p_out()))));

    auto e = with_operators<is_even, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    std::cout << "************** " << e << "\n";
    auto o = with_operators<is_odd, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    std::cout << "************** " << o << "\n";
    ASSERT_TRUE(e == 3);
    ASSERT_TRUE(o == 2);
}

TEST_F(iterate_on_esfs_class, nested_conditionals) {
    auto cond = new_cond([]() { return false; });
    auto cond2 = new_cond([]() { return true; });

    auto x =
        make_computation< BACKEND >(domain,
            grid,
            if_(cond,
                                        make_multistage(enumtype::execute< enumtype::forward >(),
                                            make_stage< iterate_on_esfs_detail::functor< 0 > >(p_in(), p_out())),
                                        if_(cond2,
                                            make_multistage(enumtype::execute< enumtype::forward >(),
                                                make_stage< iterate_on_esfs_detail::functor< 1 > >(p_in(), p_out())),
                                            make_multistage(enumtype::execute< enumtype::forward >(),
                                                make_stage< iterate_on_esfs_detail::functor< 2 > >(p_in(), p_out())))));

    auto e = with_operators<is_even, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    std::cout << "************** " << e << "\n";
    auto o = with_operators<is_odd, sum>::iterate_on_esfs<
        boost::mpl::int_< 0 >,
        typename decltype(x)::element_type::MssDescriptorArray::elements >::type::value;
    std::cout << "************** " << o << "\n";
    ASSERT_TRUE(e == 2);
    ASSERT_TRUE(o == 1);
}
