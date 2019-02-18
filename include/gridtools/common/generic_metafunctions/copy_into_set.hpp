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
#include <boost/mpl/copy.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/inserter.hpp>

namespace gridtools {
    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \ingroup mplutil
        @{
    */

    /// similar to boost::mpl::copy but it copies into an associative set container
    template <typename ToInsert, typename Seq>
    struct copy_into_set {
        typedef typename boost::mpl::copy<ToInsert,
            boost::mpl::inserter<Seq, boost::mpl::insert<boost::mpl::_1, boost::mpl::_2>>>::type type;
    };
    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
