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

namespace gridtools {
    using namespace expressions;

    using x_interval = axis<1>::full_interval;

    struct copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
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

    struct copy_twice_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out1()) = eval(in());
            eval(out2()) = eval(in());
        }
    };

    struct copy_functor_with_expression {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = eval(in() + 0.);
        }
    };

    struct copy_functor_default_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    class base_fixture : public computation_fixture<1> {
        template <class Fun, size_t... Is, class... Storages>
        void run_computation_impl(std::index_sequence<Is...>, Storages... storages) const {
            make_computation(make_multistage(execute::forward(), make_stage<Fun>(arg<Is>()...)))
                .run((arg<Is>() = storages)...);
        }

      public:
        base_fixture() : computation_fixture<1>(13, 9, 7) {}

        template <class Fun, class... Storages>
        void run_computation(Storages... storages) const {
            run_computation_impl<Fun>(std::index_sequence_for<Storages...>(), storages...);
        }

        using fun_t = std::function<double(int, int, int)>;

        fun_t input = [](int i, int j, int k) { return i * 100 + j * 10 + k; };

        fun_t shifted = [this](int i, int j, int k) { return input(i + 1, j + 1, k); };
    };

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

    struct call_proc_interface : public base_fixture {
        storage_type in = make_storage(input);
        storage_type out1 = make_storage();
        storage_type out2 = make_storage();

        void verify(storage_type actual, fun_t expected = {}) {
            if (!expected)
                expected = input;
            base_fixture::verify(make_storage(expected), actual);
        }
    };

    struct call_copy_functor_with_expression {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<copy_functor_with_expression, x_interval>::with(eval, in(), out());
        }
    };

    TEST_F(call_proc_interface, call_to_copy_functor_with_expression) {
        run_computation<call_copy_functor_with_expression>(in, out1);
        verify(out1);
    }

    struct call_copy_twice_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<copy_twice_functor, x_interval>::with(eval, in(), out1(), out2());
        }
    };

    TEST_F(call_proc_interface, call_to_copy_twice_functor) {
        run_computation<call_copy_twice_functor>(in, out1, out2);
        verify(out1);
        verify(out2);
    }

    struct call_with_offsets_copy_twice_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<copy_twice_functor, x_interval>::with(eval, in(1, 1, 0), out1(), out2());
        }
    };

    TEST_F(call_proc_interface, call_with_offsets_to_copy_twice_functor) {
        run_computation<call_with_offsets_copy_twice_functor>(in, out1, out2);
        verify(out1, shifted);
        verify(out2, shifted);
    }

    struct call_at_with_offsets_copy_twice_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<copy_twice_functor, x_interval>::at<1, 1, 0>::with(
                eval, in(), out1(-1, -1, 0), out2(-1, -1, 0)); // outs are at the original position
        }
    };

    TEST_F(call_proc_interface, call_at_with_offsets_to_copy_twice_functor) {
        run_computation<call_at_with_offsets_copy_twice_functor>(in, out1, out2);
        verify(out1, shifted);
        verify(out2, shifted);
    }

    struct call_proc_copy_functor_default_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            call_proc<copy_functor_default_interval>::with(eval, in(), out());
        }
    };

    TEST_F(call_proc_interface, call_to_copy_functor_default_interval) {
        run_computation<call_proc_copy_functor_default_interval>(in, out1);
        verify(out1);
    }

    struct call_proc_copy_functor_default_interval_with_offset_in_k {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<copy_functor_default_interval>::at<0, 0, -1>::with(eval, in(0, 0, 1), out(0, 0, 1));
        }
    };

    TEST_F(call_proc_interface, call_to_copy_functor_default_interval_with_offset_in_k) {
        run_computation<call_proc_copy_functor_default_interval_with_offset_in_k>(in, out1);
        verify(out1);
    }

    struct call_call_copy_twice_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<call_copy_twice_functor, x_interval>::with(eval, in(), out1(), out2());
        }
    };

    TEST_F(call_proc_interface, call_to_call_to_copy_twice_functor) {
        run_computation<call_call_copy_twice_functor>(in, out1, out2);
        verify(out1);
        verify(out2);
    }

    struct call_with_offsets_call_copy_twice_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<call_copy_twice_functor, x_interval>::with(eval, in(1, 1, 0), out1(), out2());
        }
    };

    TEST_F(call_proc_interface, call_with_offsets_to_call_to_copy_twice_functor) {
        run_computation<call_with_offsets_call_copy_twice_functor>(in, out1, out2);
        verify(out1, shifted);
        verify(out2, shifted);
    }

    struct call_with_offsets_call_with_offsets_copy_twice_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<call_with_offsets_copy_twice_functor, x_interval>::with(eval, in(-1, -1, 0), out1(), out2());
        }
    };

    TEST_F(call_proc_interface, call_with_offsets_to_call_with_offsets_to_copy_twice_functor) {
        run_computation<call_with_offsets_call_with_offsets_copy_twice_functor>(in, out1, out2);
        verify(out1);
        verify(out2);
    }

    struct call_with_local_variable {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out1;
        typedef inout_accessor<2, extent<>, 3> out2;
        typedef make_param_list<in, out1, out2> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            double local_in = 1;
            double local_out = -1;

            call_proc<copy_functor, x_interval>::with(eval, local_in, local_out);

            if (local_out > 0.) {
                eval(out1()) = eval(in());
            }
        }
    };

    TEST_F(call_proc_interface, call_using_local_variables) {
        run_computation<call_with_local_variable>(in, out1, out2);
        verify(out1);
    }

    struct functor_where_index_of_accessor_is_shifted_inner {
        typedef inout_accessor<0, extent<>, 3> out;
        typedef make_param_list<out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = 1.;
        }
    };

    struct functor_where_index_of_accessor_is_shifted {
        typedef inout_accessor<0, extent<>, 3> local_out;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<local_out, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<functor_where_index_of_accessor_is_shifted_inner, x_interval>::with(eval, out());
        }
    };

    struct call_with_nested_calls_and_shifted_accessor_index {
        typedef inout_accessor<0, extent<>, 3> out;
        typedef make_param_list<out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            double local_out;
            call_proc<functor_where_index_of_accessor_is_shifted, x_interval>::with(eval, local_out, out());
        }
    };

    TEST_F(call_proc_interface, call_using_local_variables_and_nested_call) {
        run_computation<call_with_nested_calls_and_shifted_accessor_index>(out1);
        verify(out1, [](int, int, int) { return 1; });
    }
} // namespace gridtools
