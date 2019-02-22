/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cassert>
#include <cstdlib>
#include <iostream>

#include <boost/mpl/vector.hpp>

#include <gridtools/stencil-composition/stencil-composition.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/
namespace gt = gridtools;

#ifdef __CUDACC__
using target_t = gt::target::cuda;
#else
using target_t = gt::target::mc;
#endif

using backend_t = gt::backend<target_t>;

using storage_info_t = gt::storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3>;
using data_store_t = gt::storage_traits<backend_t::backend_id_t>::data_store_t<double, storage_info_t>;

// These are the stencil operators that compose the multistage stencil in this test
struct copy_functor {
    using in = gt::accessor<0, gt::intent::in>;
    using out = gt::accessor<1, gt::intent::inout>;
    using param_list = gt::make_param_list<in, out>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(in());
    }
};

bool verify(data_store_t const &in, data_store_t const &out) {
    auto in_v = gt::make_host_view(in);
    auto out_v = gt::make_host_view(out);
    // check consistency

    assert(in_v.length<0>() == out_v.length<0>());
    assert(in_v.length<1>() == out_v.length<1>());
    assert(in_v.length<2>() == out_v.length<2>());

    bool success = true;
    for (int k = in_v.total_begin<2>(); k <= in_v.total_end<2>(); ++k) {
        for (int i = in_v.total_begin<0>(); i <= in_v.total_end<0>(); ++i) {
            for (int j = in_v.total_begin<1>(); j <= in_v.total_end<1>(); ++j) {
                if (in_v(i, j, k) != out_v(i, j, k)) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "in = " << in_v(i, j, k) << ", out = " << out_v(i, j, k) << std::endl;
                    success = false;
                }
            }
        }
    }
    return success;
}

int main(int argc, char **argv) {

    gt::uint_t d1, d2, d3;
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    // storage_info contains the information aboud sizes and layout of the storages to which it will be passed
    storage_info_t meta_data_{d1, d2, d3};

    // Definition of placeholders. The order does not have any semantics
    using p_in = gt::arg<0, data_store_t>;
    using p_out = gt::arg<1, data_store_t>;

    // Now we describe the itaration space. The first two dimensions
    // are described by halo_descriptors. In this case, since the
    // stencil has zero-exent, not particular care should be focused
    // on centering the iteration space to avoid put-of-bound
    // access. The third dimension is indicated with a simple size,
    // since there is not specific axis definition.
    auto grid = gt::make_grid(gt::halo_descriptor(d1), gt::halo_descriptor(d2), d3);

    data_store_t in{meta_data_, [](int i, int j, int k) { return i + j + k; }, "in"};
    data_store_t out{meta_data_, -1.0, "out"};

    auto copy = gt::make_computation<backend_t>(grid,
        p_in{} = in,
        p_out{} = out,
        gt::make_multistage(gt::execute::parallel{}, gt::make_stage<copy_functor>(p_in{}, p_out{})));

    copy.run();

    out.sync();
    in.sync();

    bool success = verify(in, out);

    if (success) {
        std::cout << "Successful\n";
    } else {
        std::cout << "Failed\n";
    }

    return !success;
};
