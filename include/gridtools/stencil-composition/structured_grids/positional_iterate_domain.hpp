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

#include "../../common/defs.hpp"
#include "../block.hpp"
#include "iterate_domain.hpp"

namespace gridtools {
    /**@brief class handling the computation of the */
    template <typename IterateDomainImpl>
    struct positional_iterate_domain : public iterate_domain<IterateDomainImpl> {
        typedef iterate_domain<IterateDomainImpl> base_t;
        typedef typename base_t::reduction_type_t reduction_type_t;
        typedef typename base_t::local_domain_t local_domain_t;

        struct array_index_t {
            typename base_t::array_index_t index;
            pos3<int_t> pos;

            auto operator[](int_t i) const GT_AUTO_RETURN(index[i]);
            auto operator[](int_t i) GT_AUTO_RETURN(index[i]);
        };

        using iterate_domain<IterateDomainImpl>::iterate_domain;

        template <int_t Step = 1>
        GT_FUNCTION void increment_i() {
            m_pos.i += Step;
            base_t::template increment_i<Step>();
        }
        template <int_t Step = 1>
        GT_FUNCTION void increment_j() {
            m_pos.j += Step;
            base_t::template increment_j<Step>();
        }
        template <int_t Step = 1>
        GT_FUNCTION void increment_k() {
            m_pos.k += Step;
            base_t::template increment_k<Step>();
        }
        GT_FUNCTION void increment_i(int_t step) {
            m_pos.i += step;
            base_t::increment_i(step);
        }
        GT_FUNCTION void increment_j(int_t step) {
            m_pos.j += step;
            base_t::increment_j(step);
        }
        GT_FUNCTION void increment_k(int_t step) {
            m_pos.k += step;
            base_t::increment_k(step);
        }

        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            static constexpr auto backend = typename base_t::iterate_domain_arguments_t::backend_ids_t{};
            static constexpr auto block_size =
                make_pos3(block_i_size(backend), block_j_size(backend), block_k_size(backend));
            m_pos.i = begin.i + block_no.i * block_size.i + pos_in_block.i;
            m_pos.j = begin.j + block_no.j * block_size.j + pos_in_block.j;
            m_pos.k = begin.k + block_no.k * block_size.k + pos_in_block.k;
            base_t::initialize(begin, block_no, pos_in_block);
        }

        GT_FUNCTION array_index_t index() const { return {base_t::index(), m_pos}; }

        GT_FUNCTION void set_index(array_index_t const &index) {
            base_t::set_index(index.index);
            m_pos = index.pos;
        }

        GT_FUNCTION int_t i() const { return m_pos.i; }

        GT_FUNCTION int_t j() const { return m_pos.j; }

        GT_FUNCTION int_t k() const { return m_pos.k; }

      private:
        pos3<int_t> m_pos;
    };
} // namespace gridtools
