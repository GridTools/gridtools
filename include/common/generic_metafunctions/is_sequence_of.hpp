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

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/quote.hpp>
#include <boost/type_traits/is_same.hpp>

/*
 * @struct is_sequence_of
 * metafunction that determines if a mpl sequence is a sequence of types determined by the filter
 * @param TSeq sequence to query
 * @param TPred filter that determines the condition
 */
template < typename TSeq, template < typename > class TPred >
struct is_sequence_of {
    typedef boost::mpl::quote1< TPred > pred_t;

    typedef typename boost::mpl::lambda< boost::mpl::not_< typename pred_t::template apply< boost::mpl::_1 > > >::type
        NegPred;

    typedef typename boost::mpl::eval_if< boost::mpl::is_sequence< TSeq >,
        boost::mpl::eval_if< boost::is_same< typename boost::mpl::find_if< TSeq, NegPred >::type,
                                 typename boost::mpl::end< TSeq >::type >,
                                              boost::mpl::true_,
                                              boost::mpl::false_ >,
        boost::mpl::false_ >::type type;

    BOOST_STATIC_CONSTANT(bool, value = (type::value));
};
