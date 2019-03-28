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

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::execute;
using namespace gridtools::expressions;

namespace call_interface_functors {

    using axis_t = axis<1>;
    using x_interval = axis_t::full_interval;
    using smaller_interval = x_interval::modify<1, -1>;

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

    struct call_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::with(eval, in());
        }
    };

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

    struct call_copy_functor_with_out_first {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor_with_out_first, x_interval>::with(eval, in());
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

    struct call_copy_functor_with_expression {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor_with_expression, x_interval>::with(eval, in());
        }
    };

    struct call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::at<1, 1, 0>::with(eval, in());
        }
    };

    struct call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::with(eval, in(1, 1, 0));
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

    struct call_copy_functor_default_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = call<copy_functor_default_interval>::with(eval, in());
        }
    };

    struct call_copy_functor_default_interval_from_smaller_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, smaller_interval) {
            eval(out()) = call<copy_functor_default_interval>::with(eval, in());
        }
    };

    struct call_copy_functor_default_interval_with_offset_in_k {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = call<copy_functor_default_interval>::at<0, 0, -1>::with(eval, in(0, 0, 1));
        }
    };

    struct call_at_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<copy_functor, x_interval>::at<-1, -1, 0>::with(eval, in(1, 1, 0));
        }
    };

    struct call_call_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_copy_functor, x_interval>::with(eval, in());
        }
    };

    struct call_call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_at_copy_functor, x_interval>::with(eval, in());
        }
    };

    struct call_call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_with_offsets_copy_functor, x_interval>::with(eval, in());
        }
    };

    struct call_at_call_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_copy_functor, x_interval>::at<1, 1, 0>::with(eval, in());
        }
    };

    struct call_at_call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_at_copy_functor, x_interval>::at<-1, -1, 0>::with(eval, in());
        }
    };

    struct call_with_offsets_call_at_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_at_copy_functor, x_interval>::with(eval, in(-1, -1, 0));
        }
    };

    struct call_at_call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_with_offsets_copy_functor, x_interval>::at<-1, -1, 0>::with(eval, in());
        }
    };

    struct call_with_offsets_call_with_offsets_copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = call<call_with_offsets_copy_functor, x_interval>::with(eval, in(-1, -1, 0));
        }
    };
} // namespace call_interface_functors

class call_interface : public testing::Test {
  protected:
    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 1;

    typedef gridtools::storage_traits<target_t>::storage_info_t<0, 3> storage_info_t;
    typedef gridtools::storage_traits<target_t>::data_store_t<float_type, storage_info_t> data_store_t;

    storage_info_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid<call_interface_functors::axis_t::axis_interval_t> grid;

    verifier verifier_;
    array<array<uint_t, 2>, 3> verifier_halos;

    data_store_t in;
    data_store_t out;

    static constexpr float_type default_value = -1;
    data_store_t reference_unchanged;
    data_store_t reference_shifted;
    data_store_t reference_smaller_interval;
    data_store_t reference_plus1;

    typedef arg<0, data_store_t> p_in;
    typedef arg<1, data_store_t> p_out;

    call_interface()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2),
          grid(make_grid(di, dj, call_interface_functors::axis_t(d3))),
#if GT_FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}},
          in(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }), out(meta_, default_value),
          reference_unchanged(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }),
          reference_shifted(meta_, [](int i, int j, int k) { return (i + 1) * 100 + (j + 1) * 10 + k; }),
          reference_smaller_interval(meta_,
              [this](int i, int j, int k) {
                  if (k > 0 && k < this->d3 - 1)
                      return (float_type)(i * 100 + j * 10 + k);
                  else
                      return default_value;
              }),
          reference_plus1(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k + 1; }) {
    }

    template <typename Computation>
    void execute_computation(Computation &comp) {
        comp.run(p_in() = in, p_out() = out);
        out.sync();
    }
};

TEST_F(call_interface, call_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(
            execute::forward(), gridtools::make_stage<call_interface_functors::call_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_with_local_variable) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_copy_functor_with_local_variable>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_plus1, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_with_local_variable2) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_copy_functor_with_local_variable2>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_plus1, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_with_out_first) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_copy_functor_with_out_first>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_with_expression) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_copy_functor_with_expression>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(
            execute::forward(), gridtools::make_stage<call_interface_functors::call_at_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_with_offsets_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_at_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_at_with_offsets_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_default_interval) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_copy_functor_default_interval>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_default_interval_from_smaller_interval) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_copy_functor_default_interval_from_smaller_interval>(
                p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_smaller_interval, out, verifier_halos));
}

TEST_F(call_interface, call_to_copy_functor_default_interval_with_offset_in_k) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_copy_functor_default_interval_with_offset_in_k>(
                p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_call_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_call_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_to_call_at_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_call_at_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_to_call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_call_with_offsets_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_call_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_at_call_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_call_at_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_at_call_at_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_with_offsets_to_call_at_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_with_offsets_call_at_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_at_to_call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_at_call_with_offsets_copy_functor>(p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

TEST_F(call_interface, call_with_offsets_to_call_with_offsets_to_copy_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_interface_functors::call_with_offsets_call_with_offsets_copy_functor>(
                p_in(), p_out())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out, verifier_halos));
}

namespace call_proc_interface_functors {
    using axis_t = axis<1>;
    using x_interval = axis_t::full_interval;

    struct copy_functor {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
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

    struct call_copy_functor_with_expression {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<copy_functor_with_expression, x_interval>::with(eval, in(), out());
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

    struct copy_functor_default_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    struct call_copy_functor_default_interval {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            call_proc<copy_functor_default_interval>::with(eval, in(), out());
        }
    };

    struct call_copy_functor_default_interval_with_offset_in_k {
        typedef in_accessor<0, extent<>, 3> in;
        typedef inout_accessor<1, extent<>, 3> out;
        typedef make_param_list<in, out> param_list;
        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            call_proc<copy_functor_default_interval>::at<0, 0, -1>::with(eval, in(0, 0, 1), out(0, 0, 1));
        }
    };

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
} // namespace call_proc_interface_functors

class call_proc_interface : public testing::Test {
  protected:
    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 1;

    typedef gridtools::storage_traits<target_t>::storage_info_t<0, 3> storage_info_t;
    typedef gridtools::storage_traits<target_t>::data_store_t<float_type, storage_info_t> data_store_t;

    storage_info_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid<call_proc_interface_functors::axis_t::axis_interval_t> grid;

    verifier verifier_;
    array<array<uint_t, 2>, 3> verifier_halos;

    data_store_t in;
    data_store_t out1;
    data_store_t out2;
    data_store_t reference_unchanged;
    data_store_t reference_shifted;
    data_store_t reference_all1;

    typedef arg<0, data_store_t> p_in;
    typedef arg<1, data_store_t> p_out1;
    typedef arg<2, data_store_t> p_out2;

    call_proc_interface()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2),
          grid(make_grid(di, dj, call_proc_interface_functors::axis_t(d3))),
#if GT_FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}},
          in(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }), out1(meta_, -5), out2(meta_, -5),
          reference_unchanged(meta_, [](int i, int j, int k) { return i * 100 + j * 10 + k; }),
          reference_shifted(meta_, [](int i, int j, int k) { return (i + 1) * 100 + (j + 1) * 10 + k; }),
          reference_all1(meta_, 1) {
    }

    template <typename Computation>
    void execute_computation(Computation &comp) {
        comp.run(/*p_in{} = in, p_out1{} = out1, p_out2{} = out2*/);
        out1.sync();
        out2.sync();
    }
};

TEST_F(call_proc_interface, call_to_copy_functor_with_expression) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_copy_functor_with_expression>(p_in(), p_out1())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
}

TEST_F(call_proc_interface, call_to_copy_twice_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        p_out2{} = out2,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_copy_twice_functor>(p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_with_offsets_to_copy_twice_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        p_out2{} = out2,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_with_offsets_copy_twice_functor>(
                p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_at_with_offsets_to_copy_twice_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        p_out2{} = out2,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_at_with_offsets_copy_twice_functor>(
                p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_to_copy_functor_default_interval) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_copy_functor_default_interval>(p_in(), p_out1())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
}

TEST_F(call_proc_interface, call_to_copy_functor_default_interval_with_offset_in_k) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_copy_functor_default_interval_with_offset_in_k>(
                p_in(), p_out1())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
}

TEST_F(call_proc_interface, call_to_call_to_copy_twice_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        p_out2{} = out2,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_call_copy_twice_functor>(
                p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_with_offsets_to_call_to_copy_twice_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        p_out2{} = out2,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_with_offsets_call_copy_twice_functor>(
                p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_shifted, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_with_offsets_to_call_with_offsets_to_copy_twice_functor) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        p_out2{} = out2,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_with_offsets_call_with_offsets_copy_twice_functor>(
                p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out2, verifier_halos));
}

TEST_F(call_proc_interface, call_using_local_variables) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_in{} = in,
        p_out1{} = out1,
        p_out2{} = out2,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_with_local_variable>(p_in(), p_out1(), p_out2())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_unchanged, out1, verifier_halos));
}

TEST_F(call_proc_interface, call_using_local_variables_and_nested_call) {
    auto comp = gridtools::make_computation<target_t>(grid,
        p_out1{} = out1,
        gridtools::make_multistage(execute::forward(),
            gridtools::make_stage<call_proc_interface_functors::call_with_nested_calls_and_shifted_accessor_index>(
                p_out1())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, reference_all1, out1, verifier_halos));
}
