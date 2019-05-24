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

#ifndef __CUDACC__
#error This is CUDA only header
#endif

#include <type_traits>
#include <utility>

#include "../../../common/array.hpp"
#include "../../../common/cuda_type_traits.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../../meta.hpp"
#include "../../caches/cache_metafunctions.hpp"
#include "../../esf_metafunctions.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../../local_domain.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template <class IterateDomainArguments>
    class iterate_domain_cuda : public iterate_domain<IterateDomainArguments> {
        GT_STATIC_ASSERT(is_iterate_domain_arguments<IterateDomainArguments>::value, GT_INTERNAL_ERROR);

        using base_t = iterate_domain<IterateDomainArguments>;

        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        using caches_t = typename local_domain_t::cache_sequence_t;
        using ij_cache_args_t = ij_cache_args<caches_t>;
        using k_cache_args_t = k_cache_args<caches_t>;
        using readwrite_args_t = compute_readwrite_args<typename IterateDomainArguments::esf_sequence_t>;

      public:
        using shared_iterate_domain_t = get_ij_cache_storage_map<caches_t,
            typename local_domain_t::max_extent_for_tmp_t,
            block_i_size(backend::cuda{}),
            block_j_size(backend::cuda{})>;

      private:
        using base_t::increment_i;
        using base_t::increment_j;

        // array storing the (i,j) position of the current thread within the block
        array<int_t, 2> m_thread_pos;
        int_t m_block_size_i;
        int_t m_block_size_j;
        shared_iterate_domain_t *GT_RESTRICT m_pshared_iterate_domain;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        template <class Arg, class T>
        static GT_FUNCTION
            std::enable_if_t<!meta::st_contains<readwrite_args_t, Arg>::value && is_texture_type<T>::value, T>
            dereference(T *ptr) {
            return __ldg(ptr);
        }
#endif
        template <class Arg, class Ptr>
        static GT_FUNCTION decltype(auto) dereference(Ptr ptr) {
            return *ptr;
        }

      public:
        static constexpr bool has_ij_caches = !meta::is_empty<ij_cache_args_t>::value;

        template <class T>
        GT_FUNCTION_DEVICE iterate_domain_cuda(T &&obj, int_t block_size_i, int_t block_size_j)
            : base_t(wstd::forward<T>(obj)), m_block_size_i(block_size_i), m_block_size_j(block_size_j) {}

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template <typename Extent>
        GT_FUNCTION bool is_thread_in_domain() const {
            return m_thread_pos[0] >= Extent::iminus::value &&
                   m_thread_pos[0] < m_block_size_i + Extent::iplus::value &&
                   m_thread_pos[1] >= Extent::jminus::value && m_thread_pos[1] < m_block_size_j + Extent::jplus::value;
        }

        GT_FUNCTION_DEVICE void set_block_pos(int_t ipos, int_t jpos) {
            m_thread_pos[0] = ipos;
            m_thread_pos[1] = jpos;
        }

        GT_FUNCTION_DEVICE void set_shared_iterate_domain_pointer(shared_iterate_domain_t *ptr) {
            m_pshared_iterate_domain = ptr;
        }

        template <class Arg, class Accessor, std::enable_if_t<meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION typename Arg::data_store_t::data_t &deref(Accessor const &acc) const {
            return host_device::at_key<Arg>(*m_pshared_iterate_domain)
                .at(m_thread_pos[0], m_thread_pos[1], this->m_color, acc);
        }

        template <class Arg, class Accessor, std::enable_if_t<!meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION decltype(auto) deref(Accessor const &acc) const {
            return dereference<Arg>(this->template get_ptr<Arg>(acc));
        }
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_cuda<IterateDomainArguments>> : std::true_type {};
} // namespace gridtools
