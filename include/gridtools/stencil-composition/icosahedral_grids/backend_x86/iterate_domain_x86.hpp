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

#include "../../iterate_domain_fwd.hpp"
#include "../iterate_domain.hpp"

namespace gridtools {

    /**
     * @brief iterate domain class for the Host backend
     */
    template <typename IterateDomainArguments>
    class iterate_domain_x86
        : public iterate_domain<iterate_domain_x86<IterateDomainArguments>, IterateDomainArguments> // CRTP
    {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_x86);
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), GT_INTERNAL_ERROR);

        typedef iterate_domain<iterate_domain_x86<IterateDomainArguments>, IterateDomainArguments> super;

      public:
        typedef typename super::strides_cached_t strides_cached_t;
        typedef typename super::local_domain_t local_domain_t;
        typedef boost::mpl::map0<> ij_caches_map_t;

        GT_FORCE_INLINE iterate_domain_x86(local_domain_t const &local_domain_) : super(local_domain_), m_strides(0) {}

        strides_cached_t &RESTRICT strides_impl() {
            assert(m_strides);
            return *m_strides;
        }

        strides_cached_t const &RESTRICT strides_impl() const {
            assert(m_strides);
            return *m_strides;
        }

        void set_strides_pointer_impl(strides_cached_t *RESTRICT strides) {
            assert(strides);
            m_strides = strides;
        }

        template <class T>
        static GT_FUNCTION auto deref_impl(T &&ptr) GT_AUTO_RETURN(*ptr);

        /**
         * caches are not currently used in host backend
         */
        template <typename IterationPolicy>
        GT_FUNCTION void slide_caches() {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }

        /**
         * caches are not currently used in host backend
         */
        template <typename IterationPolicy>
        GT_FUNCTION void flush_caches(bool) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }

        /**
         * caches are not currently used in host backend
         */
        template <typename IterationPolicy>
        GT_FUNCTION void fill_caches(bool) {
            GRIDTOOLS_STATIC_ASSERT((is_iteration_policy<IterationPolicy>::value), "error");
        }

        template <typename Extent>
        GT_FUNCTION bool is_thread_in_domain() const {
            return true;
        }

      private:
        strides_cached_t *RESTRICT m_strides;
    };

    template <typename IterateDomainArguments>
    struct is_iterate_domain<iterate_domain_x86<IterateDomainArguments>> : std::true_type {};

} // namespace gridtools
