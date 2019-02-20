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
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/vector.hpp>

#include "../defs.hpp"

namespace gridtools {

    /**
     * @struct variadic_to_vector
     * metafunction that returns a mpl vector from a pack of variadic arguments
     * This is a replacement of using type=boost::mpl::vector<Args ...>, but at the moment nvcc
     * does not properly unpack the last arg of Args... when building the vector. We can eliminate this
     * metafunction once the vector<Args...> works
     */
    template <typename... Args>
    struct variadic_to_vector;

    template <class T, typename... Args>
    struct variadic_to_vector<T, Args...> {
        typedef typename boost::mpl::push_front<typename variadic_to_vector<Args...>::type, T>::type type;
    };

    template <class T>
    struct variadic_to_vector<T> {
        typedef boost::mpl::vector1<T> type;
    };

    template <>
    struct variadic_to_vector<> {
        typedef boost::mpl::vector0<> type;
    };
} // namespace gridtools
