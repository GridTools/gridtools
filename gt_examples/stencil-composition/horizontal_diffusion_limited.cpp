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

#include "backend_select.hpp"
#include <gridtools/stencil-composition/stencil-composition.hpp>

/**
   @file
   This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
*/

namespace gt = gridtools;

using gt::accessor;
using gt::arg;
using gt::extent;
using gt::level;

// These are the stencil operators that compose the multistage stencil in this test
struct lap_function {
    typedef gt::accessor<0, gt::enumtype::inout> out;
    typedef gt::accessor<1, gt::enumtype::in, gt::extent<-1, 1, -1, 1>> in;

    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = (gt::float_type)4 * eval(in()) -
            (eval(in(1, 0, 0)) + eval(in(0, 1, 0)) + eval(in(-1, 0, 0)) + eval(in(0, -1, 0)));
    }
};

struct flx_function {

    typedef gt::accessor<0, gt::enumtype::inout> out;
    typedef gt::accessor<1, gt::enumtype::in, gt::extent<0, 1, 0, 0>> in;
    typedef gt::accessor<2, gt::enumtype::in, gt::extent<0, 1, 0, 0>> lap;

    typedef boost::mpl::vector<out, in, lap> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = eval(lap(1, 0, 0)) - eval(lap(0, 0, 0));
        if (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0) {
            eval(out()) = 0.;
        }
    }
};

struct fly_function {

    typedef gt::accessor<0, gt::enumtype::inout> out;
    typedef gt::accessor<1, gt::enumtype::in, gt::extent<0, 0, 0, 1>> in;
    typedef gt::accessor<2, gt::enumtype::in, gt::extent<0, 0, 0, 1>> lap;

    typedef boost::mpl::vector<out, in, lap> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = eval(lap(0, 1, 0)) - eval(lap(0, 0, 0));
        if (eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0) {
            eval(out()) = 0.;
        }
    }
};

struct out_function {

    typedef gt::accessor<0, gt::enumtype::inout> out;
    typedef gt::accessor<1, gt::enumtype::in> in;
    typedef gt::accessor<2, gt::enumtype::in, gt::extent<-1, 0, 0, 0>> flx;
    typedef gt::accessor<3, gt::enumtype::in, gt::extent<0, 0, -1, 0>> fly;
    typedef gt::accessor<4, gt::enumtype::in> coeff;

    typedef boost::mpl::vector<out, in, flx, fly, coeff> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) =
            eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0, 0)) + eval(fly()) - eval(fly(0, -1, 0)));
    }
};

// /*
//  * The following operators and structs are for debugging only in VERBOSE mode
//  */
// std::ostream &operator<<(std::ostream &s, lap_function const) { return s << "lap_function"; }
// std::ostream &operator<<(std::ostream &s, flx_function const) { return s << "flx_function"; }
// std::ostream &operator<<(std::ostream &s, fly_function const) { return s << "fly_function"; }
// std::ostream &operator<<(std::ostream &s, out_function const) { return s << "out_function"; }


int main(int argc, char** argv) {

    gt::uint_t halo_size = 2;

    gt::uint_t d1, d2, d3;
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    using storage_tr = gt::storage_traits<backend_t::backend_id_t>;
    using storage_info_ijk_t = storage_tr::storage_info_t<0, 3, gt::halo<2, 2, 0>>;
    using storage_type = storage_tr::data_store_t<gt::float_type, storage_info_ijk_t>;

    storage_info_ijk_t sinfo(d1,d2,d3);

    // Definition of the actual data fields that are used for input/output
    storage_type in(sinfo, "in");
    storage_type out(sinfo, "out");
    storage_type coeff(sinfo, "coeff");

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef gt::tmp_arg<0, storage_type> p_lap;
    typedef gt::tmp_arg<1, storage_type> p_flx;
    typedef gt::tmp_arg<2, storage_type> p_fly;
    typedef gt::arg<3, storage_type> p_coeff;
    typedef gt::arg<4, storage_type> p_in;
    typedef gt::arg<5, storage_type> p_out;

    gt::halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
    gt::halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

    auto grid = gt::make_grid(di, dj, d3);

    /*
      Here we do lot of stuff
      1) We pass to the intermediate representation ::run function the description
      of the stencil, which is a multi-stage stencil (mss)
      The mss includes (in order of execution) a laplacian, two fluxes which are independent
      and a final step that is the out_function
      2) The logical physical domain with the fields to use
      3) The actual grid dimensions
    */

    auto horizontal_diffusion = gt::make_computation<backend_t>
        (grid,
         p_in() = in,
         p_out() = out,
         p_coeff() = coeff,         // assign placeholders
         gt::make_multistage // mss_descriptor
         (gt::enumtype::execute<gt::enumtype::parallel, 20>(),
          define_caches(gt::cache<gt::IJ, gt::cache_io_policy::local>(p_lap(), p_flx(), p_fly())),
          gt::make_stage<lap_function>(p_lap(), p_in()), // esf_descriptor
          gt::make_independent(                          // independent_esf
                               gt::make_stage<flx_function>(p_flx(), p_in(), p_lap()),
                               gt::make_stage<fly_function>(p_fly(), p_in(), p_lap())),
          gt::make_stage<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())));

    horizontal_diffusion.run();

    out.sync();

    return 0;
}
