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

#include <utility>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../dim.hpp"
#include "../sid/blocked_dim.hpp"
#include "../sid/concept.hpp"

namespace gridtools {
    namespace cuda {
        namespace fused_mss_loop_cuda_impl_ {
            template <class Sid, class KLoop>
            struct kernel_f {
                sid::ptr_holder_type<Sid> m_ptr_holder;
                sid::strides_type<Sid> m_strides;
                KLoop k_loop;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator validator) const {
                    sid::ptr_diff_type<Sid> offset = {};
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides), blockIdx.x);
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides), blockIdx.y);
                    sid::shift(offset, sid::get_stride<dim::i>(m_strides), i_block);
                    sid::shift(offset, sid::get_stride<dim::j>(m_strides), j_block);
                    k_loop(m_ptr_holder() + offset, m_strides, wstd::move(validator));
                }
            };
        } // namespace fused_mss_loop_cuda_impl_

        template <class Composite, class KLoop>
        fused_mss_loop_cuda_impl_::kernel_f<Composite, KLoop> make_kernel(Composite &composite, KLoop k_loop) {
            return {sid::get_origin(composite), sid::get_strides(composite), std::move(k_loop)};
        }
    } // namespace cuda
} // namespace gridtools
