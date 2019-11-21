/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/positional.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/storage/traits.hpp>
#include <gridtools/tools/regression_fixture.hpp>

/**
  @file
  This file shows an implementation of the stencil in which a misaligned storage is aligned
*/

using namespace gridtools;

using alignment_test = regression_fixture<2>;

struct not_aligned {
    using acc = inout_accessor<0>;
    using out = inout_accessor<1>;
    using i_pos = in_accessor<2>;
    using param_list = make_param_list<acc, out, i_pos>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        auto *ptr = &eval(acc());
        eval(out()) = eval(i_pos()) == alignment_test::halo_size &&
                      reinterpret_cast<ptrdiff_t>(ptr) % storage::traits::alignment<storage_traits_t>;
    }
};

TEST_F(alignment_test, test) {
    auto out = make_storage<bool>();
    easy_run(not_aligned(), backend_t(), make_grid(), make_storage(), out, positional<dim::i>());
    verify(false, out);
}
