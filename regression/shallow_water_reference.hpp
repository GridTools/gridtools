/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <gridtools/common/defs.hpp>
/**
   @file
   reference implementation of the shallow_water example.
   For an exhaustive description of the shallow water problem refer to:
   http://www.mathworks.ch/moler/exm/chapters/water.pdf

*/
using namespace gridtools;
using namespace execute;

namespace {
    template <int exponent>
    static constexpr float_type pow(float_type const &x) {
        return gridtools::gt_pow<exponent>::apply(x);
    }
} // namespace

template <typename RefBackend>
struct shallow_water_reference {

    using solution_meta_t = typename RefBackend::storage_traits_t::template storage_info_t<0, 3>;
    using data_store_t = typename RefBackend::storage_traits_t::template data_store_t<float_type, solution_meta_t>;

    uint_t DimI;
    uint_t DimJ;

    solution_meta_t solution_meta;
    data_store_t u;
    data_store_t v;
    data_store_t h;
    data_store_t ux;
    data_store_t vx;
    data_store_t hx;
    data_store_t uy;
    data_store_t vy;
    data_store_t hy;

    static float_type dx() { return 1.; }
    static float_type dy() { return 1.; }
    static float_type dt() { return .02; }
    static float_type g() { return 9.81; }

    static constexpr float_type height = 2.;

    shallow_water_reference(uint_t DimI, uint_t DimJ)
        : DimI{DimI}, DimJ{DimJ}, solution_meta(DimI, DimJ, 1u), u(solution_meta, 0.0, "u"), v(solution_meta, 0.0, "v"),
          h(solution_meta, [](int i, int j, int) { return droplet_(i, j, dx(), dy(), height); }, "h"),
          ux(solution_meta, 0.0, "ux"), vx(solution_meta, 0.0, "vx"), hx(solution_meta, 0.0, "hx"),
          uy(solution_meta, 0.0, "uy"), vy(solution_meta, 0.0, "vy"), hy(solution_meta, 0.0, "hy") {}

    void iterate() {
        auto hxv = make_host_view(hx);
        auto uxv = make_host_view(ux);
        auto vxv = make_host_view(vx);
        auto uv = make_host_view(u);
        auto vv = make_host_view(v);
        auto hv = make_host_view(h);
        auto uyv = make_host_view(uy);
        auto vyv = make_host_view(vy);
        auto hyv = make_host_view(hy);

        // check if we are currently working on device or on host
        for (uint_t i = 0; i < DimI - 1; ++i)
            for (uint_t j = 0; j < DimJ - 2; ++j) {
                hxv(i, j, 0) = (hv(i + 1, j + 1, 0) + hv(i, j + 1, 0)) / 2. -
                               (uv(i + 1, j + 1, 0) - uv(i, j + 1, 0)) * (dt() / (2 * dx()));

                uxv(i, j, 0) =
                    (uv(i + 1, j + 1, 0) + uv(i, j + 1, 0)) / 2. -
                    (((pow<2>(uv(i + 1, j + 1, 0))) / hv(i + 1, j + 1, 0) + pow<2>(hv(i + 1, j + 1, 0)) * g() / 2.) -
                        (pow<2>(uv(i, j + 1, 0)) / hv(i, j + 1, 0) + pow<2>(hv(i, j + 1, 0)) * (g() / 2.))) *
                        (dt() / (2. * dx()));

                vxv(i, j, 0) = (vv(i + 1, j + 1, 0) + vv(i, j + 1, 0)) / 2. -
                               (uv(i + 1, j + 1, 0) * vv(i + 1, j + 1, 0) / hv(i + 1, j + 1, 0) -
                                   uv(i, j + 1, 0) * vv(i, j + 1, 0) / hv(i, j + 1, 0)) *
                                   (dt() / (2 * dx()));
            }

        for (uint_t i = 0; i < DimI - 2; ++i)
            for (uint_t j = 0; j < DimJ - 1; ++j) {
                hyv(i, j, 0) = (hv(i + 1, j + 1, 0) + hv(i + 1, j, 0)) / 2. -
                               (vv(i + 1, j + 1, 0) - vv(i + 1, j, 0)) * (dt() / (2 * dy()));

                uyv(i, j, 0) = (uv(i + 1, j + 1, 0) + uv(i + 1, j, 0)) / 2. -
                               (vv(i + 1, j + 1, 0) * uv(i + 1, j + 1, 0) / hv(i + 1, j + 1, 0) -
                                   vv(i + 1, j, 0) * uv(i + 1, j, 0) / hv(i + 1, j, 0)) *
                                   (dt() / (2 * dy()));

                vyv(i, j, 0) =
                    (vv(i + 1, j + 1, 0) + vv(i + 1, j, 0)) / 2. -
                    ((pow<2>(vv(i + 1, j + 1, 0)) / hv(i + 1, j + 1, 0) + pow<2>(hv(i + 1, j + 1, 0)) * g() / 2.) -
                        (pow<2>(vv(i + 1, j, 0)) / hv(i + 1, j, 0) + pow<2>(hv(i + 1, j, 0)) * (g() / 2.))) *
                        (dt() / (2. * dy()));
            }

        for (uint_t i = 1; i < DimI - 2; ++i)
            for (uint_t j = 1; j < DimJ - 2; ++j) {
                hv(i, j, 0) = hv(i, j, 0) - (uxv(i, j - 1, 0) - uxv(i - 1, j - 1, 0)) * (dt() / dx()) -
                              (vyv(i - 1, j, 0) - vyv(i - 1, j - 1, 0)) * (dt() / dy());

                uv(i, j, 0) =
                    uv(i, j, 0) -
                    (pow<2>(uxv(i, j - 1, 0)) / hxv(i, j - 1, 0) + hxv(i, j - 1, 0) * hxv(i, j - 1, 0) * ((g() / 2.)) -
                        (pow<2>(uxv(i - 1, j - 1, 0)) / hxv(i - 1, j - 1, 0) +
                            pow<2>(hxv(i - 1, j - 1, 0)) * ((g() / 2.)))) *
                        ((dt() / dx())) -
                    (vyv(i - 1, j, 0) * uyv(i - 1, j, 0) / hyv(i - 1, j, 0) -
                        vyv(i - 1, j - 1, 0) * uyv(i - 1, j - 1, 0) / hyv(i - 1, j - 1, 0)) *
                        (dt() / dy());

                vv(i, j, 0) = vv(i, j, 0) -
                              (uxv(i, j - 1, 0) * vxv(i, j - 1, 0) / hxv(i, j - 1, 0) -
                                  (uxv(i - 1, j - 1, 0) * vxv(i - 1, j - 1, 0)) / hxv(i - 1, j - 1, 0)) *
                                  ((dt() / dx())) -
                              (pow<2>(vyv(i - 1, j, 0)) / hyv(i - 1, j, 0) + pow<2>(hyv(i - 1, j, 0)) * ((g() / 2.)) -
                                  (pow<2>(vyv(i - 1, j - 1, 0)) / hyv(i - 1, j - 1, 0) +
                                      pow<2>(hyv(i - 1, j - 1, 0)) * ((g() / 2.)))) *
                                  ((dt() / dy()));
            }
    }
};
