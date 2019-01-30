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

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/stencil-functions.hpp>
#include <gridtools/tools/computation_fixture.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

struct expandable_parameters : computation_fixture<> {
    expandable_parameters() : computation_fixture<>(13, 9, 7) {}
    using storages_t = std::vector<storage_type>;

    template <class... Args>
    void run_computation(Args &&... args) const {
        gridtools::make_computation<backend_t>(expand_factor<2>(), make_grid(), std::forward<Args>(args)...).run();
    }

    void verify(storages_t const &expected, storages_t const &actual) const {
        EXPECT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i != expected.size(); ++i)
            computation_fixture<>::verify(expected[i], actual[i]);
    }
};

struct expandable_parameters_copy : expandable_parameters {

    storages_t out = {make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    storages_t in = {make_storage(-1.), make_storage(-2.), make_storage(-3.), make_storage(-4.), make_storage(-5.)};

    template <class Functor>
    void run_computation() {
        arg<0, storages_t> p_out;
        arg<1, storages_t> p_in;
        expandable_parameters::run_computation(
            p_in = in, p_out = out, make_multistage(execute<forward>(), make_stage<Functor>(p_out, p_in)));
    }

    ~expandable_parameters_copy() { verify(in, out); }
};

struct copy_functor {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> in;

    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out{}) = eval(in{});
    }
};

TEST_F(expandable_parameters_copy, copy) { run_computation<copy_functor>(); }

struct copy_functor_with_expression {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> in;

    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        // use an expression which is equivalent to a copy to simplify the check
        eval(out{}) = eval(2. * in{} - in{});
    }
};

TEST_F(expandable_parameters_copy, copy_with_expression) { run_computation<copy_functor_with_expression>(); }

struct call_proc_copy_functor {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> in;

    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        call_proc<copy_functor>::with(eval, out(), in());
    }
};

TEST_F(expandable_parameters_copy, call_proc_copy) { run_computation<call_proc_copy_functor>(); }

struct call_copy_functor {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> in;

    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = call<copy_functor>::with(eval, in());
    }
};

TEST_F(expandable_parameters_copy, call_copy) { run_computation<call_copy_functor>(); }

struct shift_functor {
    typedef accessor<0, enumtype::inout, extent<0, 0, 0, 0, -1, 0>> out;

    typedef boost::mpl::vector<out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = eval(out(0, 0, -1));
    }
};

struct call_shift_functor {
    typedef accessor<0, enumtype::inout, extent<0, 0, 0, 0, -1, 0>> out;

    typedef boost::mpl::vector<out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, axis<1>::full_interval::modify<1, 0>) {
        call_proc<shift_functor>::with(eval, out());
    }
    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, axis<1>::full_interval::first_level) {}
};

TEST_F(expandable_parameters, call_shift) {
    auto expected = [&](float_type value) { return make_storage([=](int_t i, int_t j, int_t k) { return value; }); };
    auto in = [&](float_type value) {
        return make_storage([=](int_t i, int_t j, int_t k) { return k == 0 ? value : -1; });
    };

    storages_t actual = {in(14), in(15), in(16), in(17), in(18)};
    arg<0, storages_t> plh;
    run_computation(plh = actual, make_multistage(execute<forward>(), make_stage<call_shift_functor>(plh)));
    verify({expected(14), expected(15), expected(16), expected(17), expected(18)}, actual);
}

TEST_F(expandable_parameters, caches) {
    storages_t out = {make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    auto in = make_storage(42.);

    arg<0, storages_t> p_out;
    arg<1> p_in;
    tmp_arg<1> p_tmp;
    run_computation(p_in = in,
        p_out = out,
        make_multistage(execute<forward>(),
            define_caches(cache<IJ, cache_io_policy::local>(p_tmp)),
            make_stage<copy_functor>(p_tmp, p_in),
            make_stage<copy_functor>(p_out, p_tmp)));
    verify({in, in, in, in, in}, out);
}
