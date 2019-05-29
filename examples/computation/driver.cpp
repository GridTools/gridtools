#include "interpolate_stencil.hpp"

#include <iostream>

bool verify(double weight, data_store_t const &in1, data_store_t const &in2, data_store_t const &out) {
    auto in1_v = gridtools::make_host_view<gridtools::access_mode::read_only>(in1);
    auto in2_v = gridtools::make_host_view<gridtools::access_mode::read_only>(in2);
    auto out_v = gridtools::make_host_view<gridtools::access_mode::read_only>(out);

    // check consistency
    assert(in1_v.length<0>() == out_v.length<0>());
    assert(in1_v.length<1>() == out_v.length<1>());
    assert(in1_v.length<2>() == out_v.length<2>());
    assert(in2_v.length<0>() == out_v.length<0>());
    assert(in2_v.length<1>() == out_v.length<1>());
    assert(in2_v.length<2>() == out_v.length<2>());

    bool success = true;
    for (int k = in1_v.total_begin<2>(); k <= in1_v.total_end<2>(); ++k) {
        for (int i = in1_v.total_begin<0>(); i <= in1_v.total_end<0>(); ++i) {
            for (int j = in1_v.total_begin<1>(); j <= in1_v.total_end<1>(); ++j) {
                if (weight * in1_v(i, j, k) + (1.0 - weight) * in2_v(i, j, k) - out_v(i, j, k) > 1e-8) {
                    std::cout << "error in " << i << ", " << j << ", " << k << ": "
                              << "expected = " << weight * in1_v(i, j, k) + (1.0 - weight) * in2_v(i, j, k)
                              << ", out = " << out_v(i, j, k) << ", diff = "
                              << weight * in1_v(i, j, k) + (1.0 - weight) * in2_v(i, j, k) - out_v(i, j, k)
                              << std::endl;
                    success = false;
                }
            }
        }
    }
    return success;
}
int main(int argc, char **argv) {
    unsigned int d1, d2, d3;
    const double weight = 0.4;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " dimx dimy dimz\n";
        return 1;
    } else {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
        d3 = atoi(argv[3]);
    }

    // storage_info contains the information about sizes and layout of the storages to which it will be passed
    storage_info_t meta_data_{d1, d2, d3};

    // Now we describe the iteration space. In this simple example the iteration space is just described by the full
    // grid (no particular care has to be taken to describe halo points).
    auto grid = gridtools::make_grid(d1, d2, d3);

    // Create some data stores
    data_store_t in1{meta_data_, [](int i, int j, int k) { return i + j + k; }, "in"};
    data_store_t in2{meta_data_, [](int i, int j, int k) { return 4 * i + 2 * j + k; }, "in"};
    data_store_t out{meta_data_, -1.0, "out"};

    // Use the wrapped computation
    interpolate_stencil my_stencil{grid, weight};
    my_stencil.run({in1, in2}, {out});

    out.sync();
    in1.sync();
    in2.sync();

    bool success = verify(weight, in1, in2, out);

    if (success) {
        std::cout << "Successful\n";
    } else {
        std::cout << "Failed\n";
    }

    return !success;
};
