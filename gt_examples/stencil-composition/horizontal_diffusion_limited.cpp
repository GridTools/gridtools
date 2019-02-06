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

#include <cstdlib>
#include <iostream>

#include <boost/mpl/vector.hpp>

#include <gridtools/stencil-composition/stencil-composition.hpp>

/**
   @file This file shows an implementation of the "horizontal
   diffusion" stencil, similar to the one used in COSMO since it
   implements flux-limiting
*/

namespace gt = gridtools;

namespace gt = gridtools;

// The following macros are defined by GridTools private compilation
// flags for examples, regression and unit tests. Their are not
// exported when GridTools is installed, so the user would not be
// biased by GridTools conventions.
#ifdef BACKEND_X86
using target_t = gt::target::x86;
#ifdef BACKEND_STRATEGY_NAIVE
using strategy_t = gt::strategy::naive;
#else
using strategy_t = gt::strategy::block;
#endif
#elif defined(BACKEND_MC)
using target_t = gt::target::mc;
using strategy_t = gt::strategy::block;
#elif defined(BACKEND_CUDA)
using target_t = gt::target::cuda;
using strategy_t = gt::strategy::block;
#else
#define NO_BACKEND
#endif

#ifndef NO_BACKEND
using backend_t = gt::backend<target_t, gt::grid_type::structured, strategy_t>;
#endif

// These are the stencil operators that compose the multistage stencil in this test
struct lap_function {
    using out = gt::accessor<0, gt::enumtype::inout>;
    using in = gt::accessor<1, gt::enumtype::in, gt::extent<-1, 1, -1, 1>>;

    using arg_list = gt::make_arg_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) =
            4. * eval(in()) - (eval(in(1, 0, 0)) + eval(in(0, 1, 0)) + eval(in(-1, 0, 0)) + eval(in(0, -1, 0)));
    }
};

struct flx_function {

    using out = gt::accessor<0, gt::enumtype::inout>;
    using in = gt::accessor<1, gt::enumtype::in, gt::extent<0, 1, 0, 0>>;
    using lap = gt::accessor<2, gt::enumtype::in, gt::extent<0, 1, 0, 0>>;

    using arg_list = gt::make_arg_list<out, in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        // Instead of using a temporary variable we writedirectly to
        // eval(out()) twice. This will eliminate a possible thread
        // divergenge on GPUs since we can avoid to put the `else`
        // branch below
        eval(out()) = eval(lap(1, 0, 0)) - eval(lap(0, 0, 0));
        if (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0) {
            eval(out()) = 0.;
        }
    }
};

struct fly_function {

    using out = gt::accessor<0, gt::enumtype::inout>;
    using in = gt::accessor<1, gt::enumtype::in, gt::extent<0, 0, 0, 1>>;
    using lap = gt::accessor<2, gt::enumtype::in, gt::extent<0, 0, 0, 1>>;

    using arg_list = gt::make_arg_list<out, in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        // Instead of using a temporary variable we writedirectly to
        // eval(out()) twice. This will eliminate a possible thread
        // divergenge on GPUs since we can avoid to put the `else`
        // branch below
        eval(out()) = eval(lap(0, 1, 0)) - eval(lap(0, 0, 0));
        if (eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0) {
            eval(out()) = 0.;
        }
    }
};

struct out_function {

    using out = gt::accessor<0, gt::enumtype::inout>;
    using in = gt::accessor<1, gt::enumtype::in>;
    using flx = gt::accessor<2, gt::enumtype::in, gt::extent<-1, 0, 0, 0>>;
    using fly = gt::accessor<3, gt::enumtype::in, gt::extent<0, 0, -1, 0>>;
    using coeff = gt::accessor<4, gt::enumtype::in>;

    using arg_list = gt::make_arg_list<out, in, flx, fly, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) =
            eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0, 0)) + eval(fly()) - eval(fly(0, -1, 0)));
    }
};

int main(int argc, char **argv) {

    constexpr gt::uint_t halo_size = 2;

    gt::uint_t d1, d2, d3;
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    using storage_tr = gt::storage_traits<backend_t::backend_id_t>;
    using storage_info_ijk_t = storage_tr::storage_info_t<0, 3, gt::halo<halo_size, halo_size, 0>>;
    using storage_type = storage_tr::data_store_t<double, storage_info_ijk_t>;

    // storage_info contains the information aboud sizes and layout of the storages to which it will be passed
    storage_info_ijk_t sinfo{d1, d2, d3};

    // Definition of the actual data fields that are used for input/output, instantiated using the storage_info
    storage_type in{sinfo, "in"};
    storage_type out{sinfo, "out"};
    storage_type coeff{sinfo, "coeff"};

    // Definition of placeholders. The order does not have any semantics
    using p_lap = gt::tmp_arg<0, storage_type>; // This represent a
                                                // temporary data (the
                                                // library will take
                                                // care of that and it
                                                // is not observable
                                                // by the user
    using p_flx = gt::tmp_arg<1, storage_type>;
    using p_fly = gt::tmp_arg<2, storage_type>;
    using p_coeff = gt::arg<3, storage_type>; // This is a regular placeholder to some data
    using p_in = gt::arg<4, storage_type>;
    using p_out = gt::arg<5, storage_type>;

    // Now we describe the itaration space. The frist two dimensions
    // are described with a tuple of values (minus, plus, begin, end,
    // length) begin and end, for each dimension represent the space
    // where the output data will be located in the data_stores, while
    // minus and plus indicate the number of halo points in the
    // indices before begin and after end, respectively. The length,
    // is not needed, and will be removed in future versions, but we
    // keep it for now since the data structure used is the same used
    // in the communication library and there the length is used.
    gt::halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
    gt::halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};

    // The grid represent the iteration space. The third dimension is
    // indicated here as a size and the iteration space is deduced by
    // the fact that there is not an axis definition. More ocmplex
    // third dimensions are possible but not described in this
    // example.
    auto grid = gt::make_grid(di, dj, d3);

    // Here we make the computation, specifying the backend, the gird
    // (iteration space), binding of the placeholders to the fields
    // that will not be modified during the computation, and then the
    // stencil structure
    auto horizontal_diffusion = gt::make_computation<backend_t>(grid,
        p_coeff{} = coeff, // Binding data_stores that will not change during the application
        gt::make_multistage(gt::enumtype::execute<gt::enumtype::parallel>{},
            define_caches(gt::cache<gt::IJ, gt::cache_io_policy::local>(p_lap{}, p_flx{}, p_fly{})),
            gt::make_stage<lap_function>(p_lap{}, p_in{}),
            gt::make_independent(gt::make_stage<flx_function>(p_flx{}, p_in{}, p_lap{}),
                gt::make_stage<fly_function>(p_fly{}, p_in{}, p_lap{})),
            gt::make_stage<out_function>(p_out{}, p_in{}, p_flx{}, p_fly{}, p_coeff{})));

    // The execution happens here. Here we bind the placeholders to
    // the data. This binding can change at every `run` invokation
    horizontal_diffusion.run(p_in{} = in, p_out{} = out);

    out.sync();
}
