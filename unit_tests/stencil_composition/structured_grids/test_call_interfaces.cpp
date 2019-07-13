/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

#include "test_call_interfaces.hpp"

namespace gridtools {
    struct call_interface : base_fixture {
        fun_t incremented = [this](int i, int j, int k) { return input(i, j, k) + 1; };

        template <class Fun>
        void do_test(fun_t expected = {}) const {
            storage_type out = make_storage();
            run_computation<Fun>(make_storage(input), out);
            if (!expected)
                expected = input;
            verify(make_storage(expected), out);
        }
    };

    struct copy_functor_with_add {
        typedef inout_accessor<0, extent<>, 3> out;
        typedef in_accessor<1, extent<>, 3> in1;
        typedef in_accessor<2, extent<>, 3> in2;
        typedef make_param_list<out, in1, in2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = eval(in1()) + eval(in2());
        }
    };

    // The implementation is different depending on the position of the out accessor in the callee, as the position of
    // the input accessors in the call has to be shifted when it is not in the last position.
    struct copy_functor_with_out_first {
        typedef inout_accessor<0, extent<>, 3> out;
        typedef in_accessor<1, extent<>, 3> in;
        typedef make_param_list<out, in> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    struct call_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_copy_functor) { do_test<call_copy_functor>(); }

    struct call_copy_functor_with_local_variable {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            float_type local = 1.;
            eval(out()) = call<copy_functor_with_add, x_interval>::with(eval, in(), local);
        }
    };

    TEST_F(call_interface, call_to_copy_functor_with_local_variable) {
        do_test<call_copy_functor_with_local_variable>(incremented);
    }

    struct call_copy_functor_with_local_variable2 {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            float_type local = 1.;
            eval(out()) = call<copy_functor_with_add, x_interval>::with(eval, local, in());
        }
    };

    TEST_F(call_interface, call_to_copy_functor_with_local_variable2) {
        do_test<call_copy_functor_with_local_variable2>(incremented);
    }

    struct call_copy_functor_with_out_first {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor_with_out_first, x_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_copy_functor_with_out_first) { do_test<call_copy_functor_with_out_first>(); }

    struct call_proc_copy_functor_with_expression {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor_with_expression, x_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_copy_functor_with_expression) { do_test<call_proc_copy_functor_with_expression>(); }

    struct call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::at<1, 1, 0>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_at_to_copy_functor) { do_test<call_at_copy_functor>(shifted); }

    struct call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::with(eval, in(1, 1, 0));
        }
    };

    TEST_F(call_interface, call_with_offsets_to_copy_functor) { do_test<call_with_offsets_copy_functor>(shifted); }

    struct call_at_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::at<-1, -1, 0>::with(eval, in(1, 1, 0));
        }
    };

    TEST_F(call_interface, call_at_with_offsets_to_copy_functor) { do_test<call_at_with_offsets_copy_functor>(); }

    struct call_copy_functor_default_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = call<copy_functor_default_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_copy_functor_default_interval) { do_test<call_copy_functor_default_interval>(); }

    struct call_copy_functor_default_interval_from_smaller_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval::modify<1, -1>) {
            eval(out()) = call<copy_functor_default_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_copy_functor_default_interval_from_smaller_interval) {
        do_test<call_copy_functor_default_interval_from_smaller_interval>(
            [this](int i, int j, int k) { return k > 0 && k < d3() - 1 ? input(i, j, k) : 0; });
    }

    struct call_copy_functor_default_interval_with_offset_in_k {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = call<copy_functor_default_interval>::at<0, 0, -1>::with(eval, in(0, 0, 1));
        }
    };

    TEST_F(call_interface, call_to_copy_functor_default_interval_with_offset_in_k) {
        do_test<call_copy_functor_default_interval_with_offset_in_k>();
    }

    struct call_call_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_copy_functor, x_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_call_to_copy_functor) { do_test<call_call_copy_functor>(); }

    struct call_call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_at_copy_functor, x_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_call_at_to_copy_functor) { do_test<call_call_at_copy_functor>(shifted); }

    struct call_call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_with_offsets_copy_functor, x_interval>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_to_call_with_offsets_to_copy_functor) {
        do_test<call_call_with_offsets_copy_functor>(shifted);
    }

    struct call_at_call_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_copy_functor, x_interval>::at<1, 1, 0>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_at_to_call_to_copy_functor) { do_test<call_at_call_copy_functor>(shifted); }

    struct call_at_call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_at_copy_functor, x_interval>::at<-1, -1, 0>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_at_to_call_at_to_copy_functor) { do_test<call_at_call_at_copy_functor>(); }

    struct call_with_offsets_call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_at_copy_functor, x_interval>::with(eval, in(-1, -1, 0));
        }
    };

    TEST_F(call_interface, call_with_offsets_to_call_at_to_copy_functor) {
        do_test<call_with_offsets_call_at_copy_functor>();
    }

    struct call_at_call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_with_offsets_copy_functor, x_interval>::at<-1, -1, 0>::with(eval, in());
        }
    };

    TEST_F(call_interface, call_at_to_call_with_offsets_to_copy_functor) {
        do_test<call_at_call_with_offsets_copy_functor>();
    }

    struct call_with_offsets_call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_with_offsets_copy_functor, x_interval>::with(eval, in(-1, -1, 0));
        }
    };

    TEST_F(call_interface, call_with_offsets_to_call_with_offsets_to_copy_functor) {
        do_test<call_with_offsets_call_with_offsets_copy_functor>();
    }
} // namespace gridtools
