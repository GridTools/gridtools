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
#include "direction.hpp"

/**
@file
@brief This file contains the most common predicates used for the boundary condition assignment.
The predicates identify a regoin given a @ref gridtools::direction and its data members.
*/

namespace gridtools {

    struct default_predicate {
        template < typename Direction >
        bool operator()(Direction) const {
            return true;
        }
    };

    namespace boundary_function {
        /**@brief taking the power of the component to identify the boundary

           formula:
           * compute \f$ flag_*2^{component_}\f$ , which will
           be in binary representation a serie of 0s with a 1 at position either component_, or component_ +
           n_dimensions
           (depending on wether the flag is UP or LOW).
        */
        template < typename Flag >
        GT_FUNCTION static uint_t compute_boundary_id(ushort_t const &component_, Flag flag_) {
            return (((uint_t)flag_ * (1 << component_)));
        }
    }

    /**@brief predicate returning whether I am or not at the global boundary, based on a bitmap flag which is set by the
     * @ref gridtools::partitioner_trivial*/
    template < typename Partitioner >
    struct bitmap_predicate {
        Partitioner const &m_part; // see storage/partitioner_trivial.hpp

        enum Flag { UP = 1, LOW = 8 };

        bitmap_predicate(Partitioner const &p) : m_part{p} {}

        template < sign I, sign J, sign K >
        bool operator()(direction< I, J, K >) const {
            return (m_part.at_boundary(0, ((I == minus_) ? UP : LOW))) ||
                   (m_part.at_boundary(1, ((J == minus_) ? UP : LOW))) ||
                   (m_part.at_boundary(2, ((K == minus_) ? UP : LOW)));
        }
    };
} // namespace gridtools
