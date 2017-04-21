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
#include "iterate_domain.hpp"

namespace gridtools {
    /**@brief class handling the computation of the */
    template < typename IterateDomainImpl >
    struct positional_iterate_domain : public iterate_domain< IterateDomainImpl > {
        typedef iterate_domain< IterateDomainImpl > base_t;
        typedef typename base_t::reduction_type_t reduction_type_t;
        typedef typename base_t::local_domain_t local_domain_t;

        using iterate_domain< IterateDomainImpl >::iterate_domain;

        /**@brief method for incrementing the index when moving forward along the k direction */
        template < ushort_t Coordinate, typename Execution >
        GT_FUNCTION void increment() {
            if (Coordinate == 0) {
                m_i += Execution::value;
            }
            if (Coordinate == 1) {
                m_j += Execution::value;
            }
            if (Coordinate == 2)
                m_k += Execution::value;
            base_t::template increment< Coordinate, Execution >();
        }

        /**@brief method for incrementing the index when moving forward along the k direction */
        template < ushort_t Coordinate >
        GT_FUNCTION void increment(const uint_t steps_) {
            if (Coordinate == 0) {
                m_i += steps_;
            }
            if (Coordinate == 1) {
                m_j += steps_;
            }
            if (Coordinate == 2)
                m_k += steps_;
            base_t::template increment< Coordinate >(steps_);
        }

        template < ushort_t Coordinate >
        GT_FUNCTION void initialize(uint_t const &index = 0, uint_t const &block = 0) {
            if (Coordinate == 0) {
                m_i = index;
            }
            if (Coordinate == 1) {
                m_j = index;
            }
            if (Coordinate == 2) {
                m_k = index;
            }
            base_t::template initialize< Coordinate >(index, block);
        }

        template < ushort_t Coordinate >
        GT_FUNCTION void reset_positional_index(uint_t const &lowerbound = 0) {
            if (Coordinate == 0) {
                m_i = lowerbound;
            }
            if (Coordinate == 1) {
                m_j = lowerbound;
            }
            if (Coordinate == 2) {
                m_k = lowerbound;
            }
        }

        GT_FUNCTION
        int_t i() const { return m_i; }

        GT_FUNCTION
        int_t j() const { return m_j; }

        GT_FUNCTION
        int_t k() const { return m_k; }

      private:
        int_t m_i, m_j, m_k;
    };
}
