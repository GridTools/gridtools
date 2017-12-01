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

#include <boost/mpl/count_if.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/void.hpp>

#include <boost/fusion/include/find_if.hpp>
#include <boost/fusion/include/is_sequence.hpp>

#include "../../common/defs.hpp"

#include "reduction_descriptor.hpp"

namespace gridtools {

    namespace _impl {

        template < typename = void >
        struct reduction_data {
            using reduction_type_t = notype;
            notype reduced_value() const { return {}; }
            notype initial_value() const { return {}; }
            void assign(uint_t, notype) {}
            void reduce() {}
        };

        template < typename ReductionType, typename BinOp, typename EsfDescrSequence >
        struct reduction_data< reduction_descriptor< ReductionType, BinOp, EsfDescrSequence > > {
            using reduction_type_t = ReductionType;

          private:
            BinOp m_bin_op;
            std::vector< ReductionType > m_parallel_reduced_val;
            ReductionType m_initial_value;
            ReductionType m_reduced_value;

          public:
            reduction_data(ReductionType val)
                : m_initial_value(val), m_parallel_reduced_val(omp_get_max_threads(), val) {}
            ReductionType initial_value() const { return m_initial_value; }
            ReductionType parallel_reduced_val(int elem) const { return m_parallel_reduced_val[elem]; }
            void assign(uint_t elem, ReductionType reduction_value) {
                assert(elem < m_parallel_reduced_val.size());
                m_parallel_reduced_val[elem] = m_bin_op(m_parallel_reduced_val[elem], reduction_value);
            }
            void reduce() {
                m_reduced_value = m_initial_value;
                for (auto val : m_parallel_reduced_val) {
                    m_reduced_value = m_bin_op(m_reduced_value, val);
                }
            }
            ReductionType reduced_value() const { return m_reduced_value; }
        };

        template < typename MssDescriptors >
        using has_reduction_descriptor =
            boost::mpl::count_if< MssDescriptors, mss_descriptor_is_reduction< boost::mpl::_ > >;

        template < typename MssDescriptors >
        struct get_reduction_descrpitor {
            using descriptors_t =
                boost::mpl::filter_view< MssDescriptors, mss_descriptor_is_reduction< boost::mpl::_ > >;
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size< descriptors_t >::value < 2), "Error: more than one reduction found");
            using type = typename boost::mpl::eval_if< boost::mpl::empty< descriptors_t >,
                boost::mpl::void_,
                boost::mpl::front< descriptors_t > >::type;
        };

        template < typename MssDescriptors >
        using get_reduction_data_t = reduction_data< typename get_reduction_descrpitor< MssDescriptors >::type >;
    }

    template < typename MssDescriptors >
    using reduction_type = typename _impl::get_reduction_data_t< MssDescriptors >::reduction_type_t;

    template < typename MssDescriptors,
        typename std::enable_if< _impl::has_reduction_descriptor< MssDescriptors >::value, int >::type = 0 >
    _impl::get_reduction_data_t< MssDescriptors > make_reduction_data(MssDescriptors const &src) {
        GRIDTOOLS_STATIC_ASSERT((boost::fusion::traits::is_sequence< MssDescriptors >::value), GT_INTERNAL_ERROR);
        return {boost::fusion::find_if< mss_descriptor_is_reduction< boost::mpl::_ > >(src)->get()};
    }

    template < typename MssDescriptors,
        typename std::enable_if< !_impl::has_reduction_descriptor< MssDescriptors >::value, int >::type = 0 >
    _impl::reduction_data<> make_reduction_data(MssDescriptors) {
        return {};
    }

    template < typename... T >
    struct is_reduction_data : std::false_type {};

    template < typename... T >
    struct is_reduction_data< _impl::reduction_data< T... > > : std::true_type {};
}
