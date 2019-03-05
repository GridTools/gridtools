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

#include <cassert>
#include <cmath>
#include <functional>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/location_type.hpp>

#include "neighbours_of.hpp"

class operators_repository {
    using cells = gridtools::enumtype::cells;
    using edges = gridtools::enumtype::edges;
    using vertices = gridtools::enumtype::vertices;
    using uint_t = gridtools::uint_t;

    using fun_t = std::function<double(int, int, int, int)>;

    uint_t m_d1, m_d2;

    const double PI = std::atan(1) * 4;

    template <class LocationType>
    double x(int i, int c) const {
        return (i + c * 1. / LocationType::n_colors::value) / m_d1;
    }

    double y(int j) const { return j * 1. / m_d2; }

  public:
    fun_t u = [this](int i, int c, int j, int k) {
        auto t = PI * (x<edges>(i, c) + 1.5 * y(j));
        return k + 2 * (2 + cos(t) + sin(2 * t));
    };

    fun_t edge_length = [this](int i, int c, int j, int) {
        auto t = PI * (x<edges>(i, c) + 1.5 * y(j));
        return 2.95 + (2 + cos(t) + sin(2 * t)) / 4;
    };

    fun_t edge_length_reciprocal = [this](int i, int c, int j, int k) { return 1 / edge_length(i, c, j, k); };

    fun_t cell_area_reciprocal = [this](int i, int c, int j, int) {
        auto xx = x<cells>(i, c);
        auto yy = y(j);
        return 1 / (2.53 + (2 + cos(PI * (1.5 * xx + 2.5 * yy)) + sin(2 * PI * (xx + 1.5 * yy))) / 4);
    };

    fun_t dual_area_reciprocal = [this](int i, int c, int j, int) {
        auto xx = x<vertices>(i, c);
        auto yy = y(j);
        return 1 / (1.1 + (2 + cos(PI * (1.5 * xx + yy)) + sin(1.5 * PI * (xx + 1.5 * yy))) / 4);
    };

    fun_t dual_edge_length = [this](int i, int c, int j, int) {
        auto xx = x<edges>(i, c);
        auto yy = y(j);
        return 2.2 + (2 + cos(PI * (xx + 2.5 * yy)) + sin(2 * PI * (xx + 3.5 * yy))) / 4;
    };

    fun_t dual_edge_length_reciprocal = [this](int i, int c, int j, int k) { return 1 / dual_edge_length(i, c, j, k); };

    std::function<int(int, int, int, int, int)> edge_orientation = [this](int, int, int, int, int e) {
        return e % 2 ? 1 : -1;
    };

    std::function<int(int, int, int, int, int)> orientation_of_normal = [this](int, int c, int, int, int) {
        return c == 0 ? 1 : -1;
    };

    fun_t div_u = [this](int i, int c, int j, int k) {
        double res = 0;
        int e = 0;
        for (auto &&neighbour : gridtools::neighbours_of<cells, edges>(i, c, j, k)) {
            res += orientation_of_normal(i, c, j, k, e) * neighbour.call(u) * neighbour.call(edge_length);
            ++e;
        }
        return res * cell_area_reciprocal(i, c, j, k);
    };

    fun_t curl_u = [this](int i, int c, int j, int k) {
        double res = 0;
        int e = 0;
        for (auto &&neighbour : gridtools::neighbours_of<vertices, edges>(i, c, j, k)) {
            res += edge_orientation(i, c, j, 0, e) * neighbour.call(u) * neighbour.call(dual_edge_length);
            ++e;
        }
        return res * dual_area_reciprocal(i, c, j, k);
    };

    fun_t lap = [this](int i, int c, int j, int k) {
        auto neighbours_ec = gridtools::neighbours_of<edges, cells>(i, c, j, k);
        assert(neighbours_ec.size() == 2);
        auto grad_n =
            (neighbours_ec[1].call(div_u) - neighbours_ec[0].call(div_u)) * dual_edge_length_reciprocal(i, c, j, k);

        auto neighbours_vc = gridtools::neighbours_of<edges, vertices>(i, c, j, k);
        assert(neighbours_vc.size() == 2);
        auto grad_tau =
            (neighbours_vc[1].call(curl_u) - neighbours_vc[0].call(curl_u)) * edge_length_reciprocal(i, c, j, k);
        return grad_n - grad_tau;
    };

    operators_repository(uint_t d1, uint_t d2) : m_d1(d1), m_d2(d2) {}
};
