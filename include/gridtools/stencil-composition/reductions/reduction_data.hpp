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
#include <vector>

namespace gridtools {

    template < typename MssDescriptorArray, bool HasReduction >
    struct reduction_data;

    template < typename MssDescriptorArray >
    struct reduction_data< MssDescriptorArray, false > {
        typedef notype reduction_type_t;

        reduction_data(const reduction_type_t val) {}
        constexpr reduction_type_t reduced_value() const { return 0; }
        reduction_type_t initial_value() const { return 0; }

        void assign(uint_t, const reduction_type_t &) {}
        void reduce() {}
    };

    template < typename MssDescriptorArray >
    struct reduction_data< MssDescriptorArray, true > {
        GRIDTOOLS_STATIC_ASSERT(
            (is_meta_array_of< MssDescriptorArray, is_computation_token >::value), GT_INTERNAL_ERROR);

        typedef typename boost::mpl::fold< typename MssDescriptorArray::elements,
            boost::mpl::vector0<>,
            boost::mpl::eval_if< amss_descriptor_is_reduction< boost::mpl::_2 >,
                                               boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type reduction_descriptor_seq_t;

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< reduction_descriptor_seq_t >::value == 1), "Error: more than one reduction found");

        typedef typename boost::mpl::front< reduction_descriptor_seq_t >::type reduction_descriptor_t;

        typedef typename reduction_descriptor_type< reduction_descriptor_t >::type reduction_type_t;

        typedef typename reduction_descriptor_t::bin_op_t bin_op_t;

        reduction_data(const reduction_type_t val)
            : m_initial_value(val), m_parallel_reduced_val(omp_get_max_threads(), val) {}
        const reduction_type_t &initial_value() const { return m_initial_value; }
        const reduction_type_t &parallel_reduced_val(const int elem) const { return m_parallel_reduced_val[elem]; }

        void assign(uint_t elem, const reduction_type_t &reduction_value) {
            assert(elem < m_parallel_reduced_val.size());
            m_parallel_reduced_val[elem] = bin_op_t()(m_parallel_reduced_val[elem], reduction_value);
        }

        void reduce() {
            m_reduced_value = m_initial_value;
            for (auto val : m_parallel_reduced_val) {
                m_reduced_value = bin_op_t()(m_reduced_value, val);
            }
        }
        reduction_type_t reduced_value() const { return m_reduced_value; }

      private:
        std::vector< reduction_type_t > m_parallel_reduced_val;
        reduction_type_t m_initial_value;
        reduction_type_t m_reduced_value;
    };

    template < typename T >
    struct is_reduction_data;

    template < typename MssDescriptorArray, bool HasReduction >
    struct is_reduction_data< reduction_data< MssDescriptorArray, HasReduction > > : boost::mpl::true_ {};
}
