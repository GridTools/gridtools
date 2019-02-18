/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include "../defs.hpp"
#include "variadic_typedef.hpp"
#include <boost/mpl/at.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/size.hpp>

namespace gridtools {
    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \ingroup mplutil
        @{
    */

    /*
     * converts a mpl sequence of types into a variadic_typedef of a variadic pack of types
     * Example sequence_unpacker< int,float >::type == variadic_typedef< int, float >
     */
    template <typename Seq, typename... Args>
    struct sequence_unpacker {
        GT_STATIC_ASSERT((boost::mpl::size<Seq>::value > 0 || sizeof...(Args) > 0), GT_INTERNAL_ERROR);

        template <typename Seq_>
        struct rec_unpack {
            typedef typename sequence_unpacker<typename boost::mpl::pop_front<Seq_>::type,
                Args...,
                typename boost::mpl::at_c<Seq_, 0>::type>::type type;
        };

        template <typename... Args_>
        struct get_variadic_args {
            using type = variadic_typedef<Args...>;
        };

        typedef typename boost::mpl::
            eval_if_c<(boost::mpl::size<Seq>::value > 0), rec_unpack<Seq>, get_variadic_args<Args...>>::type type;
    };
    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
