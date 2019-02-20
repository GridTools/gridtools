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

#include <boost/mpl/fold.hpp>
#include <boost/mpl/logical.hpp>
#include <boost/mpl/transform_view.hpp>
#include <type_traits>

namespace gridtools {

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \ingroup mplutil
        @{
    */
    /**
     * @struct is_there_in_sequence_if
     * return true if the predicate returns true when applied, for at least one of the elements in the Sequence
     */
    template <typename Sequence, typename Pred>
    struct is_there_in_sequence_if : boost::mpl::fold<boost::mpl::transform_view<Sequence, Pred>,
                                         std::false_type,
                                         boost::mpl::or_<boost::mpl::_1, boost::mpl::_2>>::type {};
    /** @} */
    /** @} */
    /** @} */
} // namespace gridtools
