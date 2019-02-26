/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <boost/mpl/copy.hpp>
#include <boost/mpl/inserter.hpp>

#include "../defs.hpp"

namespace gridtools {

    namespace _impl {
        struct variadic_push_back {
            template <class, class>
            struct apply;
            template <template <class...> class L, class... Ts, class T>
            struct apply<L<Ts...>, T> {
                using type = L<Ts..., T>;
            };
        };
    } // namespace _impl

    /// Helper to copy MPL sequence to a variadic typelist.
    //
    //  \tparam Src  - a type thst models MPL sequence concept
    //  \tparam Dst  - a type that is instantiation of the template with the variadic template class parameter
    //
    //  Example:
    //     copy_into_variadic<boost::mpl::vector<int, double>, std::tuple> is the same as std::tuple<int, double>
    //
    template <class Src, class Dst>
    struct lazy_copy_into_variadic : boost::mpl::copy<Src, boost::mpl::inserter<Dst, _impl::variadic_push_back>> {};

    template <class Src, class Dst>
    using copy_into_variadic =
        typename boost::mpl::copy<Src, boost::mpl::inserter<Dst, _impl::variadic_push_back>>::type;
} // namespace gridtools
