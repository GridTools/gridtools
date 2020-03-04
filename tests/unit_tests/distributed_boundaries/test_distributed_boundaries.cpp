/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/distributed_boundaries/distributed_boundaries.hpp>

#include <functional>

#include <gtest/gtest.h>
#include <mpi.h>

#include <gridtools/boundary_conditions/copy.hpp>
#include <gridtools/boundary_conditions/value.hpp>
#include <gridtools/distributed_boundaries/comm_traits.hpp>
#include <gridtools/storage/builder.hpp>

#include <gcl_select.hpp>
#include <multiplet.hpp>
#include <storage_select.hpp>
#include <timer_select.hpp>

using namespace gridtools;
using namespace std::placeholders;

constexpr int halo_size = 2;
constexpr int d1 = 6;
constexpr int d2 = 7;
constexpr int d3 = 2;

const array<halo_descriptor, 3> halos = {halo_descriptor(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
    halo_descriptor(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2),
    halo_descriptor(d3)};

const auto builder = storage::builder<storage_traits_t>.type<triplet>().halos(2, 2, 0).dimensions(d1, d2, d3);

using storage_t = decltype(builder());
using testee_t = distributed_boundaries<comm_traits<storage_t, gcl_arch_t, timer_impl_t>>;

// Returns the relative coordinates of a neighbor processor given the dimensions of a storage
int region(int index, int size) { return index < halo_size ? -1 : index >= size - halo_size ? 1 : 0; }

bool from_core(int i, int j) { return region(i, d1) == 0 and region(j, d2) == 0; }

struct distributed_boundaries_test : testing::Test {
    testee_t testee;
    int pi, pj, pk;
    int PI, PJ, PK;

    storage_t a;
    storage_t b;
    storage_t c;
    storage_t d;

    using expected_t = std::function<triplet(int, int, int)>;

    expected_t expected_a;
    expected_t expected_b;
    expected_t expected_d;

    distributed_boundaries_test()
        : testee(halos, {false, false, false}, 3, [] {
              int dims[3] = {};
              MPI_Dims_create(GCL_procs(), 3, dims);
              int period[3] = {1, 1, 1};
              MPI_Comm res;
              MPI_Cart_create(GCL_world(), 3, dims, period, false, &res);
              return res;
          }()) {
        testee.proc_grid().coords(pi, pj, pk);
        testee.proc_grid().dims(PI, PJ, PK);

        a = builder.initializer([=](int i, int j, int k) { return from_core(i, j) ? a_init(i, j, k) : triplet{}; })();
        b = builder.initializer([=](int i, int j, int k) { return from_core(i, j) ? b_init(i, j, k) : triplet{}; })();
        c = builder.initializer([=](int i, int j, int k) { return c_init(i, j, k); })();
        d = builder.initializer([=](int i, int j, int k) { return from_core(i, j) ? d_init(i, j, k) : triplet{}; })();
    }

    void expect_a(expected_t f) { expected_a = f; }
    void expect_b(expected_t f) { expected_b = f; }
    void expect_d(expected_t f) { expected_d = f; }

    ~distributed_boundaries_test() {
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d2; ++j)
                for (int k = 0; k < d3; ++k) {
                    if (expected_a) {
                        EXPECT_EQ(a->host_view()(i, j, k), expected_a(i, j, k))
                            << GCL_pid() << ": " << i << ", " << j << ", " << k;
                    }
                    if (expected_b) {
                        EXPECT_EQ(b->host_view()(i, j, k), expected_b(i, j, k))
                            << GCL_pid() << ": " << i << ", " << j << ", " << k;
                    }
                    if (expected_d) {
                        EXPECT_EQ(d->host_view()(i, j, k), expected_d(i, j, k))
                            << GCL_pid() << ": " << i << ", " << j << ", " << k;
                    }
                }
    }

    triplet a_init(int i, int j, int k) {
        return {i + pi * (d1 - 2 * halo_size) + 100,
            j + pj * (d2 - 2 * halo_size) + 100,
            k + pk * (d3 - 2 * halo_size) + 100};
    }

    triplet b_init(int i, int j, int k) const {
        return {i + pi * (d1 - 2 * halo_size) + 1000,
            j + pj * (d2 - 2 * halo_size) + 1000,
            k + pk * (d3 - 2 * halo_size) + 1000};
    }

    triplet c_init(int i, int j, int k) const {
        return {i + pi * (d1 - 2 * halo_size) + 10000,
            j + pj * (d2 - 2 * halo_size) + 10000,
            k + pk * (d3 - 2 * halo_size) + 10000};
    }

    triplet d_init(int i, int j, int k) const {
        return {i + pi * (d1 - 2 * halo_size) + 100000,
            j + pj * (d2 - 2 * halo_size) + 100000,
            k + pk * (d3 - 2 * halo_size) + 100000};
    }

    bool from_abroad(int i, int j) const {
        return (i + pi * d1 < halo_size or j + pj * d2 < halo_size or i + pi * d1 >= PI * d1 - halo_size or
                   j + pj * d2 >= PJ * d2 - halo_size) and
               testee.proc_grid().proc(region(i, d1), region(j, d2), 0) == -1;
    }
};

TEST_F(distributed_boundaries_test, boundary_only) {
    testee.boundary_only(
        bind_bc(value_boundary<triplet>(triplet{42, 42, 42}), a), bind_bc(copy_boundary(), b, _1).associate(c), d);
    expect_a([&](int i, int j, int k) {
        return from_core(i, j) ? a_init(i, j, k) : from_abroad(i, j) ? triplet{42, 42, 42} : triplet{};
    });
    expect_b([&](int i, int j, int k) {
        return from_core(i, j) ? b_init(i, j, k) : from_abroad(i, j) ? c_init(i, j, k) : triplet{};
    });
    expect_d([&](int i, int j, int k) { return from_core(i, j) ? d_init(i, j, k) : triplet{}; });
}

TEST_F(distributed_boundaries_test, exchange) {
    testee.exchange(
        bind_bc(value_boundary<triplet>(triplet{42, 42, 42}), a), bind_bc(copy_boundary(), b, _1).associate(c), d);
    expect_a([&](int i, int j, int k) { return from_abroad(i, j) ? triplet{42, 42, 42} : a_init(i, j, k); });
    expect_b([&](int i, int j, int k) { return from_abroad(i, j) ? c_init(i, j, k) : b_init(i, j, k); });
    expect_d([&](int i, int j, int k) { return from_abroad(i, j) ? triplet{} : d_init(i, j, k); });
}
