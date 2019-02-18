/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil-composition/iterate_on_esfs.hpp>

#include <type_traits>

#include <boost/mpl/vector.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/accessor.hpp>
#include <gridtools/stencil-composition/arg.hpp>
#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/make_stage.hpp>
#include <gridtools/stencil-composition/make_stencils.hpp>
#include <gridtools/storage/storage-facility.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace gridtools {
    namespace {

        template <int I>
        struct functor {
            static const int thevalue = I;
            using param_list = boost::mpl::vector<in_accessor<0>, inout_accessor<1>>;
        };

        typedef storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3> storage_info_t;
        typedef storage_traits<backend_t::backend_id_t>::data_store_t<float_type, storage_info_t> data_store_t;

        typedef arg<0, data_store_t> p_in;
        typedef arg<1, data_store_t> p_out;

        template <int I>
        using an_esf = decltype(make_stage<functor<I>>(p_in{}, p_out{}));

        template <class... Esfs>
        using an_independent = decltype(make_independent(Esfs{}...));

        template <class... Esfs>
        using an_mss = decltype(make_multistage(execute::forward(), std::declval<Esfs>()...));

        template <typename StencilOp>
        struct is_even : std::integral_constant<int, !(StencilOp::esf_function::thevalue % 2)> {};

        template <typename StencilOp>
        struct is_odd : std::integral_constant<int, !!(StencilOp::esf_function::thevalue % 2)> {};

        template <typename A, typename B>
        struct sum : std::integral_constant<int, A::value + B::value> {};

        template <typename Msses>
        using get_even = typename with_operators<is_even, sum>::iterate_on_esfs<boost::mpl::int_<0>, Msses>::type;

        template <typename Msses>
        using get_odd = typename with_operators<is_odd, sum>::iterate_on_esfs<boost::mpl::int_<0>, Msses>::type;

        using basic_t = boost::mpl::vector<an_mss<an_esf<0>, an_esf<1>, an_esf<2>, an_esf<3>, an_esf<4>>>;
        static_assert(get_even<basic_t>::value == 3, "");
        static_assert(get_odd<basic_t>::value == 2, "");

        using two_multistages_t =
            boost::mpl::vector<an_mss<an_esf<0>, an_esf<1>, an_esf<2>>, an_mss<an_esf<3>, an_esf<4>>>;
        static_assert(get_even<two_multistages_t>::value == 3, "");
        static_assert(get_odd<two_multistages_t>::value == 2, "");

        using two_multistages_independent_t =
            boost::mpl::vector<an_mss<an_esf<0>, an_independent<an_esf<1>, an_esf<2>>>, an_mss<an_esf<3>, an_esf<4>>>;
        static_assert(get_even<two_multistages_independent_t>::value == 3, "");
        static_assert(get_odd<two_multistages_independent_t>::value == 2, "");

        using just_independent_t = boost::mpl::vector<an_mss<an_independent<an_esf<1>, an_esf<2>>>, an_mss<an_esf<4>>>;
        static_assert(get_even<just_independent_t>::value == 2, "");
        static_assert(get_odd<just_independent_t>::value == 1, "");

        TEST(dummy, dummy) {}
    } // namespace
} // namespace gridtools
