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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../execution_types.hpp"
#include "../grid.hpp"
#include "../sid/concept.hpp"
#include "basic_token_execution_cuda.hpp"
#include "iterate_domain_cuda.hpp"
#include "launch_kernel.hpp"

namespace gridtools {
    namespace cuda {
        namespace fused_mss_loop_cuda_impl_ {
            template <class Grid>
            GT_FUNCTION_DEVICE auto compute_kblock(execute::forward, Grid const &grid) {
                return grid.k_min();
            };

            template <class Grid>
            GT_FUNCTION_DEVICE auto compute_kblock(execute::backward, Grid const &grid) {
                return grid.k_max();
            };

            template <class ExecutionType, class Grid>
            GT_FUNCTION_DEVICE std::enable_if_t<execute::is_parallel<ExecutionType>::value, int_t> compute_kblock(
                ExecutionType, Grid const &grid) {
                return blockIdx.z * ExecutionType::block_size + grid.k_min();
            };

            template <class Sid, class KLoop>
            struct kernel_f {
                //                GT_STATIC_ASSERT(std::is_trivially_copyable<LocalDomain>::value, GT_INTERNAL_ERROR);
                //                GT_STATIC_ASSERT(std::is_trivially_copyable<Grid>::value, GT_INTERNAL_ERROR);

                sid::ptr_holder_type<Sid> m_ptr_holder;
                sid::strides_type<Sid> m_strides;
                //                LocalDomain m_local_domain;
                KLoop k_loop;
                //                Grid m_grid;
                //                LoopIntervals m_loop_intervals;

                //                template <class P, class Dim, class Offset>
                //                GT_FUNCTION_DEVICE void shift(P &p, Offset offset) const {
                //                    sid::shift(p, sid::get_stride<Dim>(m_strides), offset);
                //                }

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator validator) const {
                    sid::ptr_diff_type<Sid> offset = {};
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides), blockIdx.x);
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides), blockIdx.y);
                    sid::shift(offset, sid::get_stride<dim::i>(m_strides), i_block);
                    sid::shift(offset, sid::get_stride<dim::j>(m_strides), j_block);
                    //                    shift<sid::blocked_dim<dim::i>>(ptr_offset, blockIdx.x);
                    //                    shift<sid::blocked_dim<dim::j>>(ptr_offset, blockIdx.y);
                    //                    shift<dim::i>(ptr_offset, iblock);
                    //                    shift<dim::j>(ptr_offset, jblock);
                    k_loop(m_ptr_holder() + offset, m_strides, wstd::move(validator));

                    /*
                    GT_FUNCTION_DEVICE iterate_domain(LocalDomain const &local_domain, int_t ipos, int_t jpos, int_t
                    kpos) : m_local_domain(local_domain), m_ptr(local_domain.m_ptr_holder()) { typename
                    LocalDomain::ptr_diff_t ptr_offset{}; shift<sid::blocked_dim<dim::i>>(ptr_offset, blockIdx.x);
                        shift<sid::blocked_dim<dim::j>>(ptr_offset, blockIdx.y);
                        shift<dim::i>(ptr_offset, ipos);
                        shift<dim::j>(ptr_offset, jpos);
                        shift<dim::k>(ptr_offset, kpos);
                        m_ptr = m_ptr + ptr_offset;
                        bind_k_caches(m_k_caches_holder, m_ptr, m_local_domain.m_strides);
                    }
*/

                    //                    iterate_domain<LocalDomain> it_domain(m_local_domain,
                    //                    m_local_domain.m_ptr_holder() + ptr_offset);
                    //                    run_functors_on_interval<ExecutionType>(it_domain, m_loop_intervals,
                    //                    wstd::move(validator));
                }
            };
        } // namespace fused_mss_loop_cuda_impl_

        template <class Composite, class KLoop>
        fused_mss_loop_cuda_impl_::kernel_f<Composite, KLoop> make_kernel(Composite &composite, KLoop k_loop) {
            return {sid::get_origin(composite), sid::get_strides(composite), std::move(k_loop)};
        }
    } // namespace cuda
} // namespace gridtools
