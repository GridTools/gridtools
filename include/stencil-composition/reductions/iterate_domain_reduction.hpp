/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

namespace gridtools {

    namespace impl {
        template < typename IterateDomainArguments, bool IsReduction >
        class iterate_domain_reduction_impl {
            typedef typename IterateDomainArguments::functor_return_type_t reduction_type_t;

          public:
            GT_FUNCTION
            iterate_domain_reduction_impl(const reduction_type_t &initial_value) {}
            GT_FUNCTION
            reduction_type_t reduction_value() const { return 0; }
        };

        template < typename IterateDomainArguments >
        class iterate_domain_reduction_impl< IterateDomainArguments, true > {
          protected:
            typedef typename IterateDomainArguments::functor_return_type_t reduction_type_t;

          public:
            GT_FUNCTION
            iterate_domain_reduction_impl(const reduction_type_t &initial_value) : m_reduced_value(initial_value) {}

            GT_FUNCTION
            reduction_type_t reduction_value() const { return m_reduced_value; }

            GT_FUNCTION
            void set_reduction_value(reduction_type_t value) { m_reduced_value = value; }

          private:
            reduction_type_t m_reduced_value;
        };
    }

    template < typename IterateDomainArguments >
    class iterate_domain_reduction
        : public impl::iterate_domain_reduction_impl< IterateDomainArguments, IterateDomainArguments::s_is_reduction > {
      public:
        typedef typename IterateDomainArguments::functor_return_type_t reduction_type_t;

        GT_FUNCTION
        iterate_domain_reduction(const reduction_type_t &initial_value)
            : impl::iterate_domain_reduction_impl< IterateDomainArguments, IterateDomainArguments::s_is_reduction >(
                  initial_value) {}
    };
}
