/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#define GT_PEDANTIC_DISABLED

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

using storage_traits_t = storage_traits<backend_t>;
using storage_info_t = storage_traits_t::storage_info_t<0, 3>;
using data_store_t = storage_traits_t::data_store_t<float_type, storage_info_t>;

struct boundary {

    int int_value;

    boundary() {}
    boundary(int ival) : int_value(ival) {}

    GT_FUNCTION
    double value() const { return 10.; }
};

struct functor1 {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(sol), GT_GLOBAL_ACCESSOR(bd));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) += eval(bd()).value() + eval(bd()).int_value;
    }
};

struct functor2 {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(sol), GT_IN_ACCESSOR(in), GT_GLOBAL_ACCESSOR(bd));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) += eval(in()) + eval(bd()).int_value;
    }
};

struct functor_with_procedure_call {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(sol), GT_GLOBAL_ACCESSOR(bd));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        call_proc<functor1>::with(eval, sol(), bd());
    }
};

struct functor1_with_assignment {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(sol), GT_GLOBAL_ACCESSOR(bd));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) = eval(bd()).value() + eval(bd()).int_value;
    }
};

struct functor_with_function_call {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(sol), GT_GLOBAL_ACCESSOR(bd));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(sol()) = call<functor1_with_assignment>::return_type<double>::with(eval, bd());
    }
};

class global_accessor_single_stage : public ::testing::Test {
  public:
    global_accessor_single_stage()
        : sinfo(10, 10, 10), sol_(sinfo, 2.), bd(20), bd_(make_global_parameter<backend_t>(bd)), di(1, 0, 1, 9, 10),
          dj(1, 0, 1, 1, 2), coords_bc(make_grid(di, dj, 2)) {}

    void check(data_store_t, float_type) {}

  protected:
    storage_info_t sinfo;
    data_store_t sol_;
    boundary bd;
    global_parameter<backend_t, boundary> bd_;

    using p_sol = arg<0, data_store_t>;
    using p_bd = arg<1, decltype(bd_)>;

    halo_descriptor di;
    halo_descriptor dj;

    grid<axis<1>::axis_interval_t> coords_bc;
};

TEST_F(global_accessor_single_stage, boundary_conditions) {
    /*****RUN 1 WITH bd int_value set to 20****/
    auto bc_eval = make_computation<backend_t>(coords_bc,
        p_sol() = sol_,
        p_bd() = bd_,
        make_multistage(execute::forward(), make_stage<functor1>(p_sol(), p_bd())));

    bc_eval.run();
    // fetch data and check
    sol_.clone_from_device();
    auto solv = make_host_view(sol_);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 20;
                }
                ASSERT_TRUE((solv(i, j, k) == value));
            }
        }
    }

    // get the configuration object from the gpu
    // modify configuration object (boundary)
    bd.int_value = 30;
    update_global_parameter(bd_, bd);

    // get the storage object from the gpu
    // modify storage object
    for (unsigned i = 0; i < 10; ++i) {
        for (unsigned j = 0; j < 10; ++j) {
            for (unsigned k = 0; k < 10; ++k) {
                solv(i, j, k) = 2.;
            }
        }
    }

    sol_.clone_to_device();

    // run again and finalize
    bc_eval.run();

    sol_.clone_from_device();
    sol_.reactivate_host_write_views();

    // check result of second run
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 30;
                }
                ASSERT_TRUE((solv(i, j, k) == value));
            }
        }
    }
}

TEST_F(global_accessor_single_stage, with_procedure_call) {
    auto bc_eval = make_computation<backend_t>(coords_bc,
        p_sol() = sol_,
        p_bd() = bd_,
        make_multistage(execute::forward(), make_stage<functor_with_procedure_call>(p_sol(), p_bd())));

    bc_eval.run();

    sol_.clone_from_device();
    auto solv = make_host_view(sol_);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 10.;
                    value += 20;
                }
                ASSERT_EQ(value, solv(i, j, k));
            }
        }
    }
}

TEST_F(global_accessor_single_stage, with_function_call) {
    auto bc_eval = make_computation<backend_t>(coords_bc,
        p_sol() = sol_,
        p_bd() = bd_,
        make_multistage(execute::forward(), make_stage<functor_with_function_call>(p_sol(), p_bd())));

    bc_eval.run();

    sol_.clone_from_device();
    auto solv = make_host_view(sol_);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                if (i > 0 && j == 1 && k < 2) {
                    double value = 10.;
                    value += 20;
                    ASSERT_EQ(value, solv(i, j, k));
                } else
                    ASSERT_EQ(2.0, solv(i, j, k));
            }
        }
    }
}

// The following will test the global accessor in a context of multiple
// stages, where global placeholders need to be remapped to local accessor
// of the various user functors
TEST(test_global_accessor, multiple_stages) {
    storage_info_t sinfo(10, 10, 10);
    data_store_t sol_(sinfo, 2.);
    data_store_t tmp_(sinfo, 2.);

    boundary bd(20);

    auto bd_ = make_global_parameter<backend_t>(bd);

    halo_descriptor di = halo_descriptor(1, 0, 1, 9, 10);
    halo_descriptor dj = halo_descriptor(1, 0, 1, 1, 2);
    auto coords_bc = make_grid(di, dj, 2);

    typedef arg<0, data_store_t> p_sol;
    typedef arg<1, data_store_t> p_tmp;
    typedef arg<2, decltype(bd_)> p_bd;

    /*****RUN 1 WITH bd int_value set to 20****/
    auto bc_eval = make_computation<backend_t>(coords_bc,
        p_sol() = sol_,
        p_tmp() = tmp_,
        p_bd() = bd_,
        make_multistage(
            execute::forward(), make_stage<functor1>(p_tmp(), p_bd()), make_stage<functor2>(p_sol(), p_tmp(), p_bd())));

    bc_eval.run();
    // fetch data and check
    sol_.clone_from_device();
    auto solv = make_host_view(sol_);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 52.;
                }
                ASSERT_TRUE(solv(i, j, k) == value);
            }
        }
    }

    // get the configuration object from the gpu
    // modify configuration object (boundary)
    bd.int_value = 30;
    update_global_parameter(bd_, bd);

    tmp_.sync();
    auto tmpv = make_host_view(tmp_);

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

    sol_.clone_to_device();
    tmp_.clone_to_device();

    // run again and finalize
    bc_eval.run();
    sol_.clone_from_device();
    sol_.reactivate_host_write_views();

    // check result of second run
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            for (int k = 0; k < 10; ++k) {
                double value = 2.;
                if (i > 0 && j == 1 && k < 2) {
                    value += 72.;
                }
                ASSERT_TRUE((solv(i, j, k) == value));
            }
        }
    }
}
