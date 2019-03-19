/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/array.hpp>
#include <gridtools/interface/layout_transformation/layout_transformation.hpp>
#include <gtest/gtest.h>

using namespace gridtools;

namespace {
    class Index {
      public:
        Index(const std::vector<uint_t> &dims, const std::vector<uint_t> &strides) : dims_(dims), strides_(strides) {
            if (dims_.size() != strides_.size())
                throw std::runtime_error("dims and strides vector have different size");
        }

        template <typename Sequence>
        uint_t operator()(const Sequence &index_set) const {
            uint_t index = 0;
            for (uint_t i = 0; i < dims_.size(); ++i) {
                if (i >= dims_[i])
                    throw std::runtime_error("index out of bounds");
                index += index_set[i] * strides_[i];
            }
            return index;
        }

        uint_t size() {
            uint_t index = 0;
            for (uint_t i = 0; i < dims_.size(); ++i) {
                index += dims_[i] * strides_[i];
            }
            return index;
        }

        uint_t size(uint_t dim) const { return dims_[dim]; }

        const std::vector<uint_t> &dims() const { return dims_; }

      private:
        std::vector<uint_t> dims_;
        std::vector<uint_t> strides_;
    };

    template <uint_t dim, typename T>
    void verify(T *expected_ptr, const Index &expected_index, T *actual_ptr, const Index &actual_index) {
        ASSERT_EQ(dim, expected_index.dims().size());

        array<size_t, dim> dims;
        std::copy(expected_index.dims().begin(), expected_index.dims().end(), dims.begin());

        for (auto index : make_hypercube_view(dims)) {
            EXPECT_EQ(expected_ptr[expected_index(index)], actual_ptr[actual_index(index)]);
        }
    }

    template <uint_t dim, typename T, typename F>
    void init(T *ptr, const Index &index, F f) {
        ASSERT_EQ(dim, index.dims().size());

        array<size_t, dim> dims;
        std::copy(index.dims().begin(), index.dims().end(), dims.begin());

        for (auto i : make_hypercube_view(dims)) {
            ptr[index(i)] = f(i);
        }
    }
} // namespace

TEST(layout_transformation, 3D_reverse_layout) {
    uint_t Nx = 4;
    uint_t Ny = 5;
    uint_t Nz = 6;

    std::vector<uint_t> dims{Nx, Ny, Nz};
    std::vector<uint_t> src_strides{1, Nx, Nx * Ny};
    std::vector<uint_t> dst_strides{Ny * Nz, Nz, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    init<3>(src, src_index, [](const array<size_t, 3> &a) { return a[0] * 100 + a[1] * 10 + a[2]; });

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    init<3>(dst, dst_index, [](const array<size_t, 3> &) { return -1; });

    gridtools::interface::transform(dst, src, dims, dst_strides, src_strides);

    verify<3>(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
}

TEST(layout_transformation, 4D_reverse_layout) {
    uint_t Nx = 4;
    uint_t Ny = 5;
    uint_t Nz = 6;
    uint_t Nw = 7;

    std::vector<uint_t> dims{Nx, Ny, Nz, Nw};
    std::vector<uint_t> src_strides{1, Nx, Nx * Ny, Nx * Ny * Nz};
    std::vector<uint_t> dst_strides{Ny * Nz * Nw, Nz * Nw, Nw, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    init<4>(src, src_index, [](const array<size_t, 4> &a) { return a[0] * 1000 + a[1] * 100 + a[2] * 10 + a[3]; });

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    init<4>(dst, dst_index, [](const array<size_t, 4> &) { return -1; });

    gridtools::interface::transform(dst, src, dims, dst_strides, src_strides);

    verify<4>(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
}

TEST(layout_transformation, 2D_reverse_layout) {
    uint_t Nx = 4;
    uint_t Ny = 5;

    std::vector<uint_t> dims{Nx, Ny};
    std::vector<uint_t> src_strides{1, Nx};
    std::vector<uint_t> dst_strides{Ny, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    init<2>(src, src_index, [](const array<size_t, 2> &a) { return a[0] * 10 + a[1]; });

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    init<2>(dst, dst_index, [](const array<size_t, 2> &) { return -1; });

    gridtools::interface::transform(dst, src, dims, dst_strides, src_strides);

    verify<2>(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
}

TEST(layout_transformation, 1D_layout_with_stride2) {
    uint_t Nx = 4;

    uint_t stride1 = 1;
    uint_t stride2 = 2;

    std::vector<uint_t> dims{Nx};
    std::vector<uint_t> src_strides{stride1};
    std::vector<uint_t> dst_strides{stride2};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    init<1>(src, src_index, [](const array<size_t, 1> &a) { return a[0]; });

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    for (uint_t i = 0; i < dst_index.size(); ++i) { // all element including the unreachable by index
        dst[i] = -1;
    }

    gridtools::interface::transform(dst, src, dims, dst_strides, src_strides);

    for (uint_t i = 0; i < dst_index.size(0); ++i) { // here we iterate only over the indexable elements!
        ASSERT_EQ(
            src[src_index(array<size_t, 1>{i})], dst[dst_index(array<size_t, 1>{i})]); // the indexable elements match
        ASSERT_EQ(-1, dst[2 * i + 1]); // the non-indexable are not touched
    }
    verify<1>(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
}

TEST(layout_transformation, one_dimension_too_many) {
    std::vector<uint_t> dims(GT_TRANSFORM_MAX_DIM + 1);
    std::vector<uint_t> src_strides(GT_TRANSFORM_MAX_DIM + 1);
    std::vector<uint_t> dst_strides(GT_TRANSFORM_MAX_DIM + 1);

    Index src_index(dims, src_strides);
    double *src = new double;

    Index dst_index(dims, dst_strides);
    double *dst = new double;

    ASSERT_ANY_THROW(gridtools::interface::transform(dst, src, dims, dst_strides, src_strides));

    delete src;
    delete dst;
}
