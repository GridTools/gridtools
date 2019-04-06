/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * test_computation.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: carlosos
 */

#ifdef NDEBUG
#undef NDEBUG
#define __WAS_DEBUG
#endif

#include <boost/fusion/include/make_vector.hpp>
#include <boost/mpl/equal.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/make_computation.hpp>
#include <gridtools/stencil_composition/make_stencils.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace positional_when_debug_test {

    using axis_t = gridtools::axis<1>;
    using grid_t = gridtools::grid<axis_t::axis_interval_t>;
    using x_interval = axis_t::get_interval<0>;

    struct test_functor {
        GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(out));

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval.i();
            eval.j();
            eval.k();
        }
    };
} // namespace positional_when_debug_test

TEST(test_make_computation, positional_when_debug) {

    using namespace gridtools;
    using namespace gridtools::execute;

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<float_type, meta_data_t> storage_t;

    typedef arg<0, storage_t> p_in;

    make_computation<backend_t>(positional_when_debug_test::grid_t(halo_descriptor{}, halo_descriptor{}, {0, 0}),
        make_multistage(execute::forward(), make_stage<positional_when_debug_test::test_functor>(p_in())));
}

#ifdef __WAS_DEBUG
#undef __WAS_DEBUG
#define NDEBUG
#endif
