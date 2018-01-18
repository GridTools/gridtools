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

#include <boost/mpl/fold.hpp>
#include "independent_esf.hpp"

namespace gridtools {

    /** This class prepare for the iteration over the esf+descriptors
        of a computation. The Unary operator is applied to each esf_descriptor
        encountered, while the Binary operator if used to "accumulate" to and
        initial value the results obtained by the unary operator. This is
        basically a map-reduce on the esf_descriptors of a computation.  The
        binary operator is assumed need to be at least associative (the order
        of the visit is the program order).

        The binary op is not actually binary, but accepts extra variadic
        arguments, of which only two will be used. The reason is to make
        boost::mpl::and_ and boost::mpl::or_ metafunctions working
        with this construct.

        \tparam UnaryOp The unary operator to apply to the esf_descriptors
        \tparam BinaryOp The binary operator for the reduction
     */
    template < template < class... > class UnaryOp, template < class... > class BinaryOp >
    struct with_operators {

      private:
        /** This class applies to binary operator to the first
            argument and the result of the application of the unary
            operator. It has a special implementation to handle
            independent_esf. This is useful to simplify the mpl::fold
            over the esf_sequence, so that we can just pass the
            placeholders.
         */
        template < class Acc, class El >
        struct apply_to_esf_sequence_elem {
            using type = typename BinaryOp< Acc, typename UnaryOp< El >::type >::type;
        };

        template < class Acc, class EsfSeq >
        struct apply_to_esf_sequence_elem< Acc, independent_esf< EsfSeq > > {
            using type = typename boost::mpl::fold< EsfSeq,
                Acc,
                apply_to_esf_sequence_elem< boost::mpl::_1, boost::mpl::_2 > >::type;
        };

        template < class Current, class MssDescriptor >
        struct apply_to_esfs {
            using type = typename boost::mpl::fold< typename MssDescriptor::esf_sequence_t,
                Current,
                apply_to_esf_sequence_elem< boost::mpl::_1, boost::mpl::_2 > >::type;
        };

      public:
        /** Given the initial value and the sequence of
            mss_descriptors (from the computation this us usually
            obtained by requesting MssDescriptorArray::elements) it
            traverses the structure to perform the visit described
            above.

            \tparam Initial The initial value of the reduction
            \tparam The sequence of mss_descriptors
         */
        template < class Initial, class MssDescriptorSeq >
        using iterate_on_esfs = typename boost::mpl::fold< MssDescriptorSeq,
            Initial,
            apply_to_esfs< boost::mpl::_1, boost::mpl::_2 > >::type;

    }; // struct with_operators
} // namespace gridtools
