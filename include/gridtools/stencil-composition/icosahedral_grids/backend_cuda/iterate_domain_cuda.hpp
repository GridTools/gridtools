/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include <type_traits>

#include "../../backend_cuda/shared_iterate_domain.hpp"
#include "../../iterate_domain_fwd.hpp"
#include "../grid_traits.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the CUDA backend
     */
    template <typename IterateDomainArguments>
    class iterate_domain_cuda
        : public iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> // CRTP
    {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_cuda);
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);

        typedef iterate_domain<iterate_domain_cuda<IterateDomainArguments>, IterateDomainArguments> super;
        typedef typename IterateDomainArguments::local_domain_t local_domain_t;

        typedef typename local_domain_t::esf_args local_domain_args_t;

      public:
        typedef typename super::grid_topology_t grid_topology_t;
        typedef typename super::strides_cached_t strides_cached_t;
        typedef typename super::iterate_domain_cache_t iterate_domain_cache_t;

        typedef shared_iterate_domain<strides_cached_t,
            typename IterateDomainArguments::max_extent_t,
            typename iterate_domain_cache_t::ij_caches_tuple_t>
            shared_iterate_domain_t;

      private:
        typedef typename super::readonly_args_indices_t readonly_args_indices_t;

        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;
        typedef typename iterate_domain_cache_t::bypass_caches_set_t bypass_caches_set_t;

      private:
        const uint_t m_block_size_i;
        const uint_t m_block_size_j;
        shared_iterate_domain_t *RESTRICT m_pshared_iterate_domain;

        using super::increment_i;
        using super::increment_j;

      public:
        template <typename Accessor>
        struct accessor_return_type {
            typedef typename super::template accessor_return_type<Accessor>::type type;
        };

        GT_FUNCTION
        explicit iterate_domain_cuda(local_domain_t const &local_domain,
            grid_topology_t const &grid_topology,
            const uint_t block_size_i,
            const uint_t block_size_j)
            : super(local_domain, grid_topology), m_block_size_i(block_size_i), m_block_size_j(block_size_j) {}

        /**
         * @brief determines whether the current (i,j) position is within the block size
         */
        template <typename Extent>
        GT_FUNCTION bool is_thread_in_domain() const {
            GRIDTOOLS_STATIC_ASSERT((is_extent<Extent>::value), GT_INTERNAL_ERROR);
            return (m_thread_pos[0] >= Extent::iminus::value &&
                    m_thread_pos[0] < ((int)m_block_size_i + Extent::iplus::value) &&
                    m_thread_pos[1] >= Extent::jminus::value &&
                    m_thread_pos[1] < ((int)m_block_size_j + Extent::jplus::value));
        }

        GT_FUNCTION
        void set_block_pos(int_t ipos, int_t jpos) {
            m_thread_pos[0] = ipos;
            m_thread_pos[1] = jpos;
        }

        GT_FUNCTION
        void set_shared_iterate_domain_pointer_impl(shared_iterate_domain_t *ptr) { m_pshared_iterate_domain = ptr; }

        GT_FUNCTION
        strides_cached_t const &RESTRICT strides_impl() const {
            //        assert((m_pshared_iterate_domain);
            return m_pshared_iterate_domain->strides();
        }
        GT_FUNCTION
        strides_cached_t &RESTRICT strides_impl() {
            //        assert((m_pshared_iterate_domain));
            return m_pshared_iterate_domain->strides();
        }

        /** @brief metafunction that determines if an arg is pointing to a field which is read only by all ESFs
         */
        template <typename Accessor>
        struct accessor_points_to_readonly_arg {

            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);

            typedef typename boost::mpl::at<local_domain_args_t,
                boost::mpl::integral_c<int, Accessor::index_t::value>>::type arg_t;

            typedef typename boost::mpl::has_key<readonly_args_indices_t,
                boost::mpl::integral_c<int, arg_index<arg_t>::value>>::type type;
        };

        /**
         * @brief metafunction that determines if an accessor has to be read from texture memory
         */
        template <typename Accessor>
        struct accessor_read_from_texture {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            typedef typename boost::mpl::and_<
                typename boost::mpl::and_<typename accessor_points_to_readonly_arg<Accessor>::type,
                    typename boost::mpl::not_<typename boost::mpl::has_key<bypass_caches_set_t,
                        static_uint<Accessor::index_t::value>>::type>::type>::type,
                typename boost::is_arithmetic<typename accessor_return_type<Accessor>::type>>::type type;
        };

        /** @brief return a value that was cached
         * specialization where cache is not explicitly disabled by user
         */
        template <uint_t Color, typename ReturnType, typename Accessor>
        GT_FUNCTION
            typename boost::disable_if<boost::mpl::has_key<bypass_caches_set_t, static_uint<Accessor::index_t::value>>,
                ReturnType>::type
            get_cache_value_impl(Accessor const &_accessor) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            //        assert(m_pshared_iterate_domain);
            // retrieve the ij cache from the fusion tuple and access the element required give the current thread
            // position within the block and the offsets of the accessor
            return m_pshared_iterate_domain->template get_ij_cache<static_uint<Accessor::index_t::value>>()
                .template at<Color>(m_thread_pos, _accessor);
        }

        /** @brief return a value that was cached
         * specialization where cache is explicitly disabled by user
         */
        template <typename ReturnType, typename Accessor>
        GT_FUNCTION
            typename boost::enable_if<boost::mpl::has_key<bypass_caches_set_t, static_uint<Accessor::index_t::value>>,
                ReturnType>::type
            get_cache_value_impl(Accessor const &_accessor) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            return super::template get_value<Accessor, void * RESTRICT>(
                _accessor, aux::get_data_pointer(super::m_local_domain, _accessor));
        }

        /** @brief return a the value in memory pointed to by an accessor
         * specialization where the accessor points to an arg which is readonly for all the ESFs in all MSSs
         * Value is read via texture system
         */
        template <typename ReturnType, typename Accessor, typename StorageType>
        GT_FUNCTION typename boost::enable_if<typename accessor_read_from_texture<Accessor>::type, ReturnType>::type
        get_value_impl(StorageType *RESTRICT storage_pointer, int_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
#if __CUDA_ARCH__ >= 350
            // on Kepler use ldg to read directly via read only cache
            return __ldg(storage_pointer + pointer_offset);
#else
            return *(storage_pointer + pointer_offset);
#endif
        }

        /** @brief return a the value in memory pointed to by an accessor
         * specialization where the accessor points to an arg which is not readonly for all the ESFs in all MSSs
         */
        template <typename ReturnType, typename Accessor, typename StorageType>
        GT_FUNCTION typename boost::disable_if<typename accessor_read_from_texture<Accessor>::type, ReturnType>::type
        get_value_impl(StorageType *RESTRICT storage_pointer, int_t pointer_offset) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
            return *(storage_pointer + pointer_offset);
        }

        // kcaches not yet implemented
        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }
        template <typename IterationPolicy, typename Grid>
        GT_FUNCTION void flush_caches(const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "error");
        }
        template <typename IterationPolicy, typename Grid>
        GT_FUNCTION void fill_caches(const int_t klevel, Grid const &grid) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
            GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "error");
        }
        template <typename IterationPolicy>
        GT_FUNCTION void final_flush() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }
        template <typename IterationPolicy>
        GT_FUNCTION void begin_fill() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }

      private:
        // array storing the (i,j) position of the current thread within the block
        array<int, 2> m_thread_pos;
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_cuda<IterateDomainArguments>> : std::true_type {};

} // namespace gridtools
