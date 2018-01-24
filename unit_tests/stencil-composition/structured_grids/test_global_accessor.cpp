/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#define PEDANTIC_DISABLED

#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <storage/storage-facility.hpp>
#include <stencil-composition/stencil-functions/stencil-functions.hpp>

#include "backend_select.hpp"

using namespace gridtools;
using namespace enumtype;

using storage_traits_t = typename backend_t::storage_traits_t;
using storage_info_t = storage_traits_t::storage_info_t< 0, 3 >;
using data_store_t = storage_traits_t::data_store_t< float_type, storage_info_t >;

struct boundary {

    int int_value;

    boundary() {}
    boundary(int ival) : int_value(ival) {}

    GT_FUNCTION
    double value() const { return 10.; }
};

struct functor1 {
    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 > > sol;
    typedef global_accessor< 1 > bd;

    typedef boost::mpl::vector< sol, bd > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(sol()) += eval(bd()).value() + eval(bd()).int_value;
    }
};

struct functor2 {
    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 > > sol;
    typedef accessor< 1, enumtype::inout, extent< 0, 0, 0, 0 > > in;
    typedef global_accessor< 2, enumtype::inout > bd;

    typedef boost::mpl::vector< sol, in, bd > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(sol()) += eval(in()) + eval(bd()).int_value;
    }
};

struct functor_with_procedure_call {
    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 > > sol;
    typedef global_accessor< 1 > bd;

    typedef boost::mpl::vector< sol, bd > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        call_proc< functor1 >::with(eval, sol(), bd());
    }
};

struct functor1_with_assignment {
    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 > > sol;
    typedef global_accessor< 1 > bd;

    typedef boost::mpl::vector< sol, bd > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(sol()) = eval(bd()).value() + eval(bd()).int_value;
    }
};

struct functor_with_function_call {
    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 > > sol;
    typedef global_accessor< 1 > bd;

    typedef boost::mpl::vector< sol, bd > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(sol()) = call< functor1_with_assignment >::with(eval, bd());
    }
};

class global_accessor_single_stage : public ::testing::Test {
  public:
    global_accessor_single_stage()
        : sinfo(10, 10, 10), sol_(sinfo, 2.), bd(20), bd_(backend_t::make_global_parameter(bd)), di(1, 0, 1, 9, 10),
          dj(1, 0, 1, 1, 2), coords_bc(make_grid(di, dj, 2)), domain(sol_, bd_) {}

    void check(data_store_t field, float_type value) {}

  protected:
    storage_info_t sinfo;
    data_store_t sol_;
    boundary bd;
    decltype(backend_t::make_global_parameter(bd)) bd_;

    using p_sol = arg< 0, data_store_t >;
    using p_bd = arg< 1, decltype(bd_) >;

    halo_descriptor di;
    halo_descriptor dj;

    grid< axis< 1 >::axis_interval_t > coords_bc;

    aggregator_type< boost::mpl::vector< p_sol, p_bd > > domain;
};

TEST_F(global_accessor_single_stage, boundary_conditions) {
    /*****RUN 1 WITH bd int_value set to 20****/
    auto bc_eval = make_computation< backend_t >(
        domain, coords_bc, make_multistage(execute< forward >(), make_stage< functor1 >(p_sol(), p_bd())));

    bc_eval->ready();
    bc_eval->steady();
    bc_eval->run();
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
    backend_t::update_global_parameter(bd_, bd);

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
    bc_eval->run();

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
    bc_eval->finalize();
}

TEST_F(global_accessor_single_stage, with_procedure_call) {
    auto bc_eval = make_computation< backend_t >(domain,
        coords_bc,
        make_multistage(execute< forward >(), make_stage< functor_with_procedure_call >(p_sol(), p_bd())));

    bc_eval->ready();
    bc_eval->steady();
    bc_eval->run();

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

    bc_eval->finalize();
}

TEST_F(global_accessor_single_stage, with_function_call) {
    auto bc_eval = make_computation< backend_t >(domain,
        coords_bc,
        make_multistage(execute< forward >(), make_stage< functor_with_function_call >(p_sol(), p_bd())));

    bc_eval->ready();
    bc_eval->steady();
    bc_eval->run();

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

    bc_eval->finalize();
}

// The following will test the global accessor in a context of multiple
// stages, where global placeholders need to be remapped to local accessor
// of the various user functors
TEST(test_global_accessor, multiple_stages) {
    storage_info_t sinfo(10, 10, 10);
    data_store_t sol_(sinfo, 2.);
    data_store_t tmp_(sinfo, 2.);

    boundary bd(20);

    auto bd_ = backend_t::make_global_parameter(bd);

    halo_descriptor di = halo_descriptor(1, 0, 1, 9, 10);
    halo_descriptor dj = halo_descriptor(1, 0, 1, 1, 2);
    auto coords_bc = make_grid(di, dj, 2);

    typedef arg< 0, data_store_t > p_sol;
    typedef arg< 1, data_store_t > p_tmp;
    typedef arg< 2, decltype(bd_) > p_bd;

    aggregator_type< boost::mpl::vector< p_sol, p_tmp, p_bd > > domain(sol_, tmp_, bd_);

    /*****RUN 1 WITH bd int_value set to 20****/
    auto bc_eval = make_computation< backend_t >(domain,
        coords_bc,
        make_multistage(execute< forward >(),
                                                     make_stage< functor1 >(p_tmp(), p_bd()),
                                                     make_stage< functor2 >(p_sol(), p_tmp(), p_bd())));

    bc_eval->ready();
    bc_eval->steady();
    bc_eval->run();
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
    backend_t::update_global_parameter(bd_, bd);

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
    bc_eval->run();
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
    bc_eval->finalize();
}
