/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cassert>

#include <gridtools/storage/builder.hpp>
#include <type_traits>

namespace gridtools {

    struct v2e {};
    struct e2v {};

    template <class StorageTraits, class FloatType>
    class structured_unstructured_mesh {
        int m_nx, m_ny, m_nz;

        constexpr auto v2e_initializer() const {
            return [nx = m_nx, ny = m_ny](int vertex, int neighbor) {
                if (neighbor >= max_v2e_neighbors)
                    return -1;
                assert(neighbor >= 0);
                assert(vertex >= 0 && vertex < nx * ny);
                int const nxedges = (nx - 1) * ny;
                int const nyedges = nx * (ny - 1);
                int i = vertex % nx;
                int j = vertex / nx;
                if (i > 0 && neighbor-- == 0)
                    return (i - 1) + (nx - 1) * j;
                if (i < nx - 1 && neighbor-- == 0)
                    return i + (nx - 1) * j;
                if (j > 0 && neighbor-- == 0)
                    return nxedges + i + nx * (j - 1);
                if (j < ny - 1 && neighbor-- == 0)
                    return nxedges + i + nx * j;
                if (i < nx - 1 && j > 0 && neighbor-- == 0)
                    return nxedges + nyedges + i + (nx - 1) * (j - 1);
                if (i > 0 && j < ny - 1 && neighbor-- == 0)
                    return nxedges + nyedges + (i - 1) + (nx - 1) * j;
                return -1;
            };
        }

        constexpr auto e2v_initializer() const {
            return [nx = m_nx, ny = m_ny](int edge, int neighbor) {
                assert(neighbor > 0);
                if (neighbor >= max_e2v_neighbors)
                    return -1;
                int const nxedges = (nx - 1) * ny;
                int const nyedges = nx * (ny - 1);
                [[maybe_unused]] int const nxyedges = (nx - 1) * (ny - 1);
                assert(edge >= 0 && edge < nxedges + nyedges + nxyedges);
                if (edge < nxedges) {
                    int i = edge % (nx - 1);
                    int j = edge / (nx - 1);
                    if (neighbor == 1)
                        i += 1;
                    return i + nx * j;
                }
                edge -= nxedges;
                if (edge < nyedges) {
                    int i = edge % nx;
                    int j = edge / nx;
                    if (neighbor == 1)
                        j += 1;
                    return i + nx * j;
                }
                edge -= nyedges;
                assert(edge < nxyedges);
                int i = edge % (nx - 1);
                int j = edge / (nx - 1);
                if (neighbor == 1)
                    j += 1;
                else
                    i += 1;
                return i + nx * j;
            };
        }

      public:
        static constexpr int max_v2e_neighbors = 6;
        static constexpr int max_e2v_neighbors = 2;

        constexpr structured_unstructured_mesh(int nx, int ny, int nz) : m_nx(nx), m_ny(ny), m_nz(nz) {}

        constexpr int nvertices() const { return m_nx * m_ny; }
        constexpr int nedges() const {
            int nxedges = (m_nx - 1) * m_ny;
            int nyedges = m_nx * (m_ny - 1);
            int nxyedges = (m_nx - 1) * (m_ny - 1);
            return nxedges + nyedges + nxyedges;
        }
        constexpr int nlevels() const { return m_nz; }

        template <class T = FloatType, class Init, class... Dims, std::enable_if_t<!std::is_integral_v<Init>, int> = 0>
        auto make_storage(Init const &init, Dims... dims) const {
            return storage::builder<StorageTraits>.dimensions(dims...).template type<T>().initializer(init).build();
        }

        template <class T = FloatType,
            class... Dims,
            std::enable_if_t<std::conjunction_v<std::is_integral<Dims>...>, int> = 0>
        auto make_storage(Dims... dims) const {
            return make_storage<T>([](int, int) { return T(); }, dims...);
        }

        template <class T = FloatType, class... Args>
        auto make_const_storage(Args &&...args) const {
            return make_storage<T const>(std::forward<Args>(args)...);
        }

        auto v2e_table() const {
            return storage::builder<StorageTraits>.dimensions(nvertices(), max_v2e_neighbors).template type<int>().initializer(v2e_initializer()).build();
        }

        auto e2v_table() const {
            return storage::builder<StorageTraits>.dimensions(nedges(), max_e2v_neighbors).template type<int>().initializer(e2v_initializer()).build();
        }
    };

} // namespace gridtools
