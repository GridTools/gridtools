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

#include <boost/mpl/size.hpp>

namespace gridtools {

    /**
     * data structure that holds data members of the iterate domain that must be stored in shared memory.
     *
     * @tparam Strides strides cached type
     * @tparam IJCaches fusion map of <Arg, ij_cache_storage>
     */
    template <class Strides, class IJCaches, size_t = boost::mpl::size<IJCaches>::value>
    struct shared_iterate_domain {
        Strides m_strides;
        IJCaches m_ij_caches;
    };

    template <class Strides, class IJCaches>
    struct shared_iterate_domain<Strides, IJCaches, 0> {
        Strides m_strides;
    };
} // namespace gridtools
