/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "independent_esf.hpp"
#include <boost/mpl/fold.hpp>

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
    template <template <class...> class UnaryOp, template <class...> class BinaryOp>
    struct with_operators {

      private:
        /** This class applies to binary operator to the first
            argument and the result of the application of the unary
            operator. It has a special implementation to handle
            independent_esf. This is useful to simplify the mpl::fold
            over the esf_sequence, so that we can just pass the
            placeholders.
         */
        template <class Acc, class El>
        struct apply_to_esf_sequence_elem {
            using type = typename BinaryOp<Acc, typename UnaryOp<El>::type>::type;
        };

        template <class Acc, class EsfSeq>
        struct apply_to_esf_sequence_elem<Acc, independent_esf<EsfSeq>> {
            using type = typename boost::mpl::
                fold<EsfSeq, Acc, apply_to_esf_sequence_elem<boost::mpl::_1, boost::mpl::_2>>::type;
        };

        template <class Current, class MssDescriptor>
        struct apply_to_esfs {
            using type = typename boost::mpl::fold<typename MssDescriptor::esf_sequence_t,
                Current,
                apply_to_esf_sequence_elem<boost::mpl::_1, boost::mpl::_2>>::type;
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
        template <class Initial, class MssDescriptorSeq>
        using iterate_on_esfs =
            typename boost::mpl::fold<MssDescriptorSeq, Initial, apply_to_esfs<boost::mpl::_1, boost::mpl::_2>>::type;

    }; // struct with_operators
} // namespace gridtools
