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

#include <vector>

#include "../../common/defs.hpp"
#include "../../common/hypercube_iterator.hpp"
#include "../../common/layout_map_metafunctions.hpp"
#include "../../common/make_array.hpp"
#include "../../common/tuple_util.hpp"
#include "../../storage/storage-facility.hpp"
#include "layout_transformation_config.hpp"
#include "layout_transformation_helper.hpp"

namespace gridtools {
    namespace impl {
        template <typename DataType>
        void transform_openmp_loop(DataType *dst,
            DataType *src,
            const std::vector<uint_t> &dims,
            const std::vector<uint_t> &dst_strides,
            const std::vector<uint_t> &src_strides) {

            if (dims.size() > GT_TRANSFORM_MAX_DIM)
                throw std::runtime_error("Reached compile time GT_TRANSFORM_MAX_DIM in layout transformation. Increase "
                                         "the value for higher dimensional transformations.");

            using dummy_layout_map =
                default_layout_map_t<GT_TRANSFORM_MAX_DIM>; // not used since we pass strides directly

            using storage_info = gridtools::storage_info_interface<0, dummy_layout_map>;
            auto a_dims = impl::vector_to_dims_array<GT_TRANSFORM_MAX_DIM>(dims);
            auto a_dst_strides = impl::vector_to_strides_array<GT_TRANSFORM_MAX_DIM>(dst_strides);
            auto a_src_strides = impl::vector_to_strides_array<GT_TRANSFORM_MAX_DIM>(src_strides);

            storage_info si_dst(a_dims, a_dst_strides);
            storage_info si_src(a_dims, a_src_strides);

            array<size_t, GT_TRANSFORM_MAX_DIM - 3> outer_dims;
            for (size_t i = 0; i < GT_TRANSFORM_MAX_DIM - 3; ++i)
                outer_dims[i] = a_dims[3 + i];

            for (auto &&outer : make_hypercube_view(outer_dims)) {
                uint_t size_i = a_dims[0]; // because ICC 17 complains otherwise...
                uint_t size_j = a_dims[1];
                uint_t size_k = a_dims[2];
#pragma omp parallel for collapse(3)
                for (int i = 0; i < size_i; ++i)
                    for (int j = 0; j < size_j; ++j)
                        for (int k = 0; k < size_k; ++k) {
                            auto index = tuple_util::push_front(tuple_util::convert_to<array, int>(outer), i, j, k);
                            dst[si_dst.index(index)] = src[si_src.index(index)];
                        }
            }
        }
    } // namespace impl
} // namespace gridtools
