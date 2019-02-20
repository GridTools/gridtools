/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <boost/mpl/advance.hpp>
#include <boost/mpl/erase.hpp>
#include <boost/mpl/insert.hpp>

namespace gridtools {
    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \ingroup mplutil
        @{
    */

    /*
     * Replace in a sequence the element in the given position by another element
     */
    template <typename Seq_, typename Pos, typename Elem>
    struct replace {
        typedef typename boost::mpl::advance<typename boost::mpl::begin<Seq_>::type, boost::mpl::int_<Pos::value>>::type
            iter_t;
        typedef typename boost::mpl::insert<Seq_, iter_t, Elem>::type inserted_seq;

        typedef typename boost::mpl::advance<typename boost::mpl::begin<inserted_seq>::type,
            boost::mpl::int_<Pos::value + 1>>::type iter2_t;

        typedef typename boost::mpl::erase<inserted_seq, iter2_t>::type type;
    };
    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
