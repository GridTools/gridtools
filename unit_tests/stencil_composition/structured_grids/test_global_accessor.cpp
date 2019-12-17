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

#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/stencil_composition/global_parameter.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace cartesian;

const auto make_storage = storage::builder<storage_traits_t>.type<float_type>().dimensions(10, 10, 10).value(2);

struct boundary {

    int int_value;

    boundary() {}
    boundary(int ival) : int_value(ival) {}

    GT_FUNCTION
    double value() const { return 10.; }
};

struct functor1 {
    typedef inout_accessor<0> sol;
    typedef in_accessor<1> bd;

    typedef make_param_list<sol, bd> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) += eval(bd()).value() + eval(bd()).int_value;
    }
};

struct functor2 {
    typedef inout_accessor<0> sol;
    typedef in_accessor<1> in;
    typedef in_accessor<2> bd;

    typedef make_param_list<sol, in, bd> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) += eval(in()) + eval(bd()).int_value;
    }
};

struct functor_with_procedure_call {
    typedef inout_accessor<0> sol;
    typedef in_accessor<1> bd;

    typedef make_param_list<sol, bd> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        call_proc<functor1>::with(eval, sol(), bd());
    }
};

struct functor1_with_assignment {
    typedef inout_accessor<0> sol;
    typedef in_accessor<1> bd;

    typedef make_param_list<sol, bd> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) = eval(bd()).value() + eval(bd()).int_value;
    }
};

struct functor_with_function_call {
    typedef inout_accessor<0> sol;
    typedef in_accessor<1> bd;

    typedef make_param_list<sol, bd> param_list;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) = call<functor1_with_assignment>::return_type<double>::with(eval, bd());
    }
};

class global_accessor_single_stage : public ::testing::Test {
  public:
    global_accessor_single_stage()
        : bd(20), bd_(make_global_parameter(bd)), di(1, 0, 1, 9, 10), dj(1, 0, 1, 1, 2),
          coords_bc(make_grid(di, dj, 2)) {}

    boundary bd;
    global_parameter<boundary> bd_;

    halo_descriptor di;
    halo_descriptor dj;

    decltype(make_grid(halo_descriptor(), halo_descriptor(), 0)) coords_bc;
};

TEST_F(global_accessor_single_stage, boundary_conditions) {
    auto sol = make_storage();
    /*****RUN 1 WITH bd int_value set to 20****/
    auto bc_eval = [&] { easy_run(functor1(), backend_t(), coords_bc, sol, bd_); };
    bc_eval();
    // fetch data and check
    auto solv = sol->host_view();
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 20;
                }
                EXPECT_EQ(value, solv(i, j, k));
            }
        }
    }

    // get the configuration object from the gpu
    // modify configuration object (boundary)
    bd.int_value = 30;
    bd_ = make_global_parameter(bd);

    // get the storage object from the gpu
    // modify storage object
    for (unsigned i = 0; i < 10; ++i) {
        for (unsigned j = 0; j < 10; ++j) {
            for (unsigned k = 0; k < 10; ++k) {
                solv(i, j, k) = 2.;
            }
        }
    }

    // run again and finalize
    bc_eval();

    solv = sol->host_view();

    // check result of second run
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 30;
                }
                EXPECT_EQ(value, solv(i, j, k));
            }
        }
    }
}

TEST_F(global_accessor_single_stage, with_procedure_call) {
    auto sol = make_storage();
    easy_run(functor_with_procedure_call(), backend_t(), coords_bc, sol, bd_);

    auto solv = sol->host_view();
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 20;
                }
                EXPECT_EQ(value, solv(i, j, k));
            }
        }
    }
}

TEST_F(global_accessor_single_stage, with_function_call) {
    auto sol = make_storage();
    easy_run(functor_with_function_call(), backend_t(), coords_bc, sol, bd_);

    auto solv = sol->host_view();
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                if (i > 0 && j == 1 && k < 2) {
                    double value = 10.;
                    value += 20;
                    EXPECT_EQ(value, solv(i, j, k));
                } else
                    EXPECT_EQ(2, solv(i, j, k));
            }
        }
    }
}

// The following will test the global accessor in a context of multiple
// stages, where global placeholders need to be remapped to local accessor
// of the various user functors
TEST(test_global_accessor, multiple_stages) {
    auto sol = make_storage();
    auto tmp = make_storage();

    boundary bd(20);

    auto bd_ = make_global_parameter(bd);

    halo_descriptor di = halo_descriptor(1, 0, 1, 9, 10);
    halo_descriptor dj = halo_descriptor(1, 0, 1, 1, 2);
    auto coords_bc = make_grid(di, dj, 2);

    /*****RUN 1 WITH bd int_value set to 20****/
    auto bc_eval = [&] {
        run(
            [](auto sol, auto tmp, auto bd) {
                return execute_parallel().stage(functor1(), tmp, bd).stage(functor2(), sol, tmp, bd);
            },
            backend_t(),
            coords_bc,
            sol,
            tmp,
            bd_);
    };

    bc_eval();
    // fetch data and check
    auto solv = sol->host_view();
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 52.;
                }
                EXPECT_EQ(value, solv(i, j, k));
            }
        }
    }

    // get the configuration object from the gpu
    // modify configuration object (boundary)
    bd.int_value = 30;
    bd_ = make_global_parameter(bd);

    auto tmpv = tmp->host_view();

    // get the storage object from the gpu
    // modify storage object
    for (unsigned i = 0; i < 10; ++i) {
        for (unsigned j = 0; j < 10; ++j) {
            for (unsigned k = 0; k < 10; ++k) {
                tmpv(i, j, k) = 2.;
                solv(i, j, k) = 2.;
            }
        }
    }

    // run again
    bc_eval();

    solv = sol->host_view();

    // check result of second run
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 72.;
                }
                EXPECT_EQ(value, solv(i, j, k));
            }
        }
    }
}
