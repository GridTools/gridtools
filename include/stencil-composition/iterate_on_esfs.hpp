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

namespace gridtools {

    /** This class prepare for the iteration over the esf+descriptors
        of a computation. The Unary operator is applied to each esf_descriptor
        encountered, while the Binary operator if used to "accumulate" to and
        initial value the results obtained by the unary operator. This is
        basically a map-reduce on the esf_descriptors of a computation.  The
        binary operator is assumed need to be at least associative (the order
        of the visit is the program order).

        \tparam UnaryOp The unary opertor to apply to the esf_descriptors
        \tparam BinaryOp The binaty operator for the reduction
     */
    template < template < typename > class UnaryOp, template < typename, typename > class BinaryOp >
    struct with_operators {

    private:
        /** This class applies to binary operator to the first
            argument and the resullt of the application of the unary
            operator. It has a special implementation to handle
            independent_esf. This is useful to simplify the mpl::fold
            over the esf_sequence, so that we can just pass the
            placeholders.
         */
        struct compose {
            template < typename A, typename B >
            struct apply {
                using type = typename BinaryOp< A, typename UnaryOp< B >::type >::type;
            };

            template < typename A, typename... EsfList >
            struct apply< A, independent_esf< EsfList... > > {
                using list = typename independent_esf< EsfList... >::esf_list;
                using type = typename boost::mpl::fold< list,
                                                        A,
                                                        typename compose::template apply< boost::mpl::_1, boost::mpl::_2 > >::type;
            };
        };

        template < typename Current, typename MssDescriptor >
        struct apply_to_esfs {
            using type = typename boost::mpl::fold< typename MssDescriptor::esf_sequence_t,
                                                    Current,
                                                    typename compose::template apply< boost::mpl::_1, boost::mpl::_2 > >::type;
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
        template < typename Initial,
                   typename MssDescriptorSeq >
        struct iterate_on_esfs {
            typedef typename boost::mpl::fold< MssDescriptorSeq,
                                               Initial,
                                               apply_to_esfs< boost::mpl::_1, boost::mpl::_2 > >::type
            type;
        };

        /** Specialization for when conditionals are used.
         */
        template < typename Initial,
                   typename MssArray1,
                   typename MssArray2,
                   typename Tag >
        struct iterate_on_esfs< Initial, condition< MssArray1, MssArray2, Tag > > {
            using temp = typename iterate_on_esfs< Initial, MssArray1 >::type;
            using type = typename iterate_on_esfs< temp, MssArray2 >::type;
        };
    }; // struct with_operators
} // namespace gridtools
