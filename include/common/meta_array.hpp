/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
/*
 * meta_array.h
 *
 *  Created on: Feb 20, 2015
 *      Author: carlosos
 */

#pragma once
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/fold.hpp>
#include "../common/defs.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#ifdef CXX11_ENABLED
#include "meta_array_generator.hpp"
#else
#include "meta_array_generator_cxx03.hpp"
#endif

namespace gridtools {

    /**
     * @brief wrapper class around a sequence of types. The goal of the class is to identify that a type is an array of
     * types that
     * fulfil a predicate
     * (without having to inspect each element of the sequence)
     */
    template < typename Sequence, typename TPred >
    struct meta_array {
        typedef Sequence sequence_t;
        BOOST_STATIC_ASSERT((boost::mpl::is_sequence< sequence_t >::value));

        // check that predicate returns true for all elements
        typedef typename boost::mpl::fold< sequence_t,
            boost::mpl::true_,
            boost::mpl::and_< boost::mpl::_1, typename TPred::template apply< boost::mpl::_2 > > >::type
            is_array_of_pred;

        BOOST_STATIC_ASSERT((is_array_of_pred::value));

        typedef sequence_t elements;
    };

    // first order
    template < typename Sequence1, typename Sequence2, typename Cond, typename TPred >
    struct meta_array< condition< Sequence1, Sequence2, Cond >, TPred > {
        typedef Sequence1 sequence1_t;
        typedef Sequence2 sequence2_t;
        typedef condition< sequence1_t, sequence2_t, Cond > elements;
    };

    // type traits for meta_array
    template < typename T >
    struct is_meta_array : boost::mpl::false_ {};

    template < typename Sequence, typename TPred >
    struct is_meta_array< meta_array< Sequence, TPred > > : boost::mpl::true_ {};

    template < typename T, template < typename > class pred >
    struct is_meta_array_of : boost::mpl::false_ {};

    template < typename Sequence, typename pred, template < typename > class pred_query >
    struct is_meta_array_of< meta_array< Sequence, pred >, pred_query > {
        typedef typename boost::is_same< boost::mpl::quote1< pred_query >, pred >::type type;
        BOOST_STATIC_CONSTANT(bool, value = (type::value));
    };

} // namespace gridtools
