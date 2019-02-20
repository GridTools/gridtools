/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/generic_metafunctions/is_there_in_sequence_if.hpp>

#include <type_traits>

#include <boost/mpl/vector.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>

using namespace gridtools;

TEST(is_there_in_sequence_if, is_there_in_sequence_if) {
    typedef boost::mpl::vector<int, float, char> seq_t;

    GT_STATIC_ASSERT((is_there_in_sequence_if<seq_t, std::is_same<boost::mpl::_, char>>::value), "ERROR");
    GT_STATIC_ASSERT((is_there_in_sequence_if<seq_t, std::is_same<boost::mpl::_, int>>::value), "ERROR");
    GT_STATIC_ASSERT((!is_there_in_sequence_if<seq_t, std::is_same<boost::mpl::_, long>>::value), "ERROR");
}
