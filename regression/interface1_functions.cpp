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
#include <gridtools/tools/regression_fixture.hpp>

#include "horizontal_diffusion_repository.hpp"

using namespace gridtools;

struct lap_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;

    using arg_list = boost::mpl::vector<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = 4. * eval(in()) - (eval(in(-1, 0)) + eval(in(0, -1)) + eval(in(0, 1)) + eval(in(1, 0)));
    }
};

struct flx_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 2, -1, 1>>;

    using arg_list = boost::mpl::vector<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
#ifdef FUNCTIONS_MONOLITHIC
        float_type _x_ = 4. * eval(in()) - (eval(in(-1, 0)) + eval(in(0, -1)) + eval(in(0, 1)) + eval(in(1, 0)));
        float_type _y_ = 4. * eval(in(1, 0)) - (eval(in(0, 0)) + eval(in(1, -1)) + eval(in(1, 1)) + eval(in(2, 0)));
#else
#ifdef FUNCTIONS_PROCEDURES
        float_type _x_;
        call_proc<lap_function>::at<0, 0, 0>::with(eval, _x_, in());
        float_type _y_;
        call_proc<lap_function>::at<1, 0, 0>::with(eval, _y_, in());
#else
#ifdef FUNCTIONS_PROCEDURES_OFFSETS
        float_type _x_;
        call_proc<lap_function>::with(eval, _x_, in());
        float_type _y_;
        call_proc<lap_function>::with(eval, _y_, in(1, 0, 0));
#else
#ifdef FUNCTIONS_OFFSETS
        float_type _x_ = call<lap_function>::with(eval, in(0, 0, 0));
        float_type _y_ = call<lap_function>::with(eval, in(1, 0, 0));
#else
        float_type _x_ = call<lap_function>::at<0, 0, 0>::with(eval, in());
        float_type _y_ = call<lap_function>::at<1, 0, 0>::with(eval, in());
#endif
#endif
#endif
#endif
        eval(out()) = _y_ - _x_;
        eval(out()) = eval(out()) * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0.0 : eval(out());
    }
};

struct fly_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 2>>;

    using arg_list = boost::mpl::vector<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
#ifdef FUNCTIONS_MONOLITHIC
        float_type _x_ =
            4. * eval(in()) - (eval(in(-1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0)) + eval(in(1, 0, 0)));
        float_type _y_ =
            4. * eval(in(0, 1, 0)) - (eval(in(-1, 1, 0)) + eval(in(0, 0, 0)) + eval(in(0, 2, 0)) + eval(in(1, 1, 0)));
#else
#ifdef FUNCTIONS_PROCEDURES
        float_type _x_;
        call_proc<lap_function>::at<0, 0, 0>::with(eval, _x_, in());
        float_type _y_;
        call_proc<lap_function>::at<0, 1, 0>::with(eval, _y_, in());
#else
#ifdef FUNCTIONS_PROCEDURES_OFFSETS
        float_type _x_;
        call_proc<lap_function>::with(eval, _x_, in());
        float_type _y_;
        call_proc<lap_function>::with(eval, _y_, in(0, 1, 0));
#else
#ifdef FUNCTIONS_OFFSETS
        float_type _x_ = call<lap_function>::with(eval, in(0, 0, 0));
        float_type _y_ = call<lap_function>::with(eval, in(0, 1, 0));
#else
        float_type _x_ = call<lap_function>::at<0, 0, 0>::with(eval, in());
        float_type _y_ = call<lap_function>::at<0, 1, 0>::with(eval, in());
#endif
#endif
#endif
#endif
        eval(out()) = _y_ - _x_;
        eval(out()) = eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0 ? 0.0 : eval(out());
    }
};

struct out_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1>;
    using flx = in_accessor<2, extent<-1, 0, 0, 0>>;
    using fly = in_accessor<3, extent<0, 0, -1, 0>>;
    using coeff = in_accessor<4>;

    using arg_list = boost::mpl::vector<out, in, flx, fly, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) =
            eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0, 0)) + eval(fly()) - eval(fly(0, -1, 0)));
    }
};

#ifdef FUNCTIONS_MONOLITHIC
#define FTESTNAME(x) HorizontalDiffusionFunctionsMONOLITHIC
#endif

#ifdef FUNCTIONS_CALL
#define FTESTNAME(x) HorizontalDiffusionFunctionsCALL
#endif

#ifdef FUNCTIONS_OFFSETS
#define FTESTNAME(x) HorizontalDiffusionFunctionsOFFSETS
#endif

#ifdef FUNCTIONS_PROCEDURES
#define FTESTNAME(x) HorizontalDiffusionFunctionsPROCEDURES
#endif

#ifdef FUNCTIONS_PROCEDURES_OFFSETS
#define FTESTNAME(x) HorizontalDiffusionFunctionsPROCEDURESOFFSETS
#endif

using FTESTNAME(x) = regression_fixture<2>;

TEST_F(FTESTNAME(x), Test) {
    tmp_arg<0> p_flx;
    tmp_arg<1> p_fly;
    arg<1> p_coeff;
    arg<2> p_in;
    arg<3> p_out;

    auto out = make_storage();

    horizontal_diffusion_repository repo(d1(), d2(), d3());

    make_computation(p_in = make_storage(repo.in),
        p_out = out,
        p_coeff = make_storage(repo.coeff),
        make_multistage(enumtype::execute<enumtype::forward>(),
            define_caches(cache<IJ, cache_io_policy::local>(p_flx, p_fly)),
            make_independent(make_stage<flx_function>(p_flx, p_in), make_stage<fly_function>(p_fly, p_in)),
            make_stage<out_function>(p_out, p_in, p_flx, p_fly, p_coeff)))
        .run();

    verify(make_storage(repo.out), out);
}
