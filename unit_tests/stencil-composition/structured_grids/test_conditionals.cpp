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

#include <gridtools/stencil-composition/conditionals/if_.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

namespace gridtools {
    namespace {
        template <uint_t Id>
        struct functor {
            using p_dummy = inout_accessor<0>;
            using param_list = make_param_list<p_dummy>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(p_dummy()) = Id;
            }
        };

        struct stencil_composition : computation_fixture<> {
            stencil_composition() : computation_fixture<>(1, 1, 1) {}
        };

        TEST_F(stencil_composition, conditionals) {
            auto dummy = make_storage();
            make_computation(p_0 = dummy,
                if_([] { return false; },
                    make_multistage(execute::forward(), make_stage<functor<0>>(p_0)),
                    if_([] { return true; },
                        make_multistage(execute::forward(), make_stage<functor<1>>(p_0)),
                        make_multistage(execute::forward(), make_stage<functor<2>>(p_0)))))
                .run();

            verify(make_storage(1.), dummy);
        }
    } // namespace
} // namespace gridtools