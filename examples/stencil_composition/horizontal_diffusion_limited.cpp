/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdlib>
#include <iostream>

#include <gridtools/stencil_composition/stencil_composition.hpp>

/**
   @file This file shows an implementation of the "horizontal
   diffusion" stencil, similar to the one used in COSMO since it
   implements flux-limiting
*/

namespace gt = gridtools;

#ifdef __CUDACC__
using backend_t = gt::backend::cuda;
#else
using backend_t = gt::backend::mc;
#endif

// These are the stencil operators that compose the multistage stencil in this test
struct lap_function {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(out), GT_IN_ACCESSOR(in, gt::extent<-1, 1, -1, 1>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) =
            4. * eval(in()) - (eval(in(1, 0, 0)) + eval(in(0, 1, 0)) + eval(in(-1, 0, 0)) + eval(in(0, -1, 0)));
    }
};

struct flx_function {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(out),
        GT_IN_ACCESSOR(in, gt::extent<0, 1, 0, 0>),
        GT_IN_ACCESSOR(lap, gt::extent<0, 1, 0, 0>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        // Instead of using a temporary variable we write directly to
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
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(out),
        GT_IN_ACCESSOR(in, gt::extent<0, 0, 0, 1>),
        GT_IN_ACCESSOR(lap, gt::extent<0, 0, 0, 1>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
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
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(out),
        GT_IN_ACCESSOR(in),
        GT_IN_ACCESSOR(flx, gt::extent<-1, 0, 0, 0>),
        GT_IN_ACCESSOR(fly, gt::extent<0, 0, -1, 0>),
        GT_IN_ACCESSOR(coeff));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
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

    using storage_tr = gt::storage_traits<backend_t>;
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
        gt::make_multistage(gt::execute::parallel{},
            define_caches(gt::cache<gt::cache_type::ij, gt::cache_io_policy::local>(p_lap{}, p_flx{}, p_fly{})),
            gt::make_stage<lap_function>(p_lap{}, p_in{}),
            gt::make_independent(gt::make_stage<flx_function>(p_flx{}, p_in{}, p_lap{}),
                gt::make_stage<fly_function>(p_fly{}, p_in{}, p_lap{})),
            gt::make_stage<out_function>(p_out{}, p_in{}, p_flx{}, p_fly{}, p_coeff{})));

    // The execution happens here. Here we bind the placeholders to
    // the data. This binding can change at every `run` invokation
    horizontal_diffusion.run(p_in{} = in, p_out{} = out);

    out.sync();
}
