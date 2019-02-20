/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "boost/mpl/contains.hpp"
#include "boost/mpl/set.hpp"
#include "boost/mpl/size.hpp"
#include "boost/mpl/vector.hpp"
#include "gtest/gtest.h"
#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/copy_into_set.hpp>
#include <gridtools/common/host_device.hpp>

using namespace gridtools;

template <int>
struct myt {};

namespace {
    GT_FUNCTION bool test_all_elements_unique() {
        typedef boost::mpl::vector<myt<0>, myt<1>> my_vec1;
        typedef boost::mpl::vector<myt<2>, myt<3>> my_vec2;

        typedef boost::mpl::set<my_vec1, my_vec2> set_of_vecs;

        typedef typename boost::mpl::
            fold<set_of_vecs, boost::mpl::set0<>, copy_into_set<boost::mpl::_2, boost::mpl::_1>>::type result;

        GT_STATIC_ASSERT((boost::mpl::contains<result, myt<0>>::type::value == true), "is not in set");
        GT_STATIC_ASSERT((boost::mpl::contains<result, myt<1>>::type::value == true), "is not in set");
        GT_STATIC_ASSERT((boost::mpl::contains<result, myt<2>>::type::value == true), "is not in set");
        GT_STATIC_ASSERT((boost::mpl::contains<result, myt<3>>::type::value == true), "is not in set");
        GT_STATIC_ASSERT((boost::mpl::size<result>::type::value == 4), "set has wrong size");
        return true;
    }

    GT_FUNCTION bool test_repeating_element() {
        typedef boost::mpl::vector<myt<0>, myt<1>> my_vec1;
        typedef boost::mpl::vector<myt<2>, myt<0>> my_vec2;

        typedef boost::mpl::set<my_vec1, my_vec2> set_of_vecs;

        typedef typename boost::mpl::
            fold<set_of_vecs, boost::mpl::set0<>, copy_into_set<boost::mpl::_2, boost::mpl::_1>>::type result;

        GT_STATIC_ASSERT((boost::mpl::contains<result, myt<0>>::type::value == true), "is not in set");
        GT_STATIC_ASSERT((boost::mpl::contains<result, myt<1>>::type::value == true), "is not in set");
        GT_STATIC_ASSERT((boost::mpl::contains<result, myt<2>>::type::value == true), "is not in set");
        GT_STATIC_ASSERT((boost::mpl::size<result>::type::value == 3), "set has wrong size");
        return true;
    }
} // namespace

TEST(copy_into_set, all_elements_unique) { ::test_all_elements_unique(); }

TEST(copy_into_set, repeating_element) { ::test_repeating_element(); }
