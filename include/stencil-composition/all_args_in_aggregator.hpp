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

#include <boost/mpl/and.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/contains.hpp>

namespace gridtools {

    namespace _impl {
        template < typename Aggregator >
        struct investigate_esf {
            template < typename Agg >
            struct investigate_placeholder {
                template < typename CR, typename Plc >
                struct apply {
                    using type = typename boost::mpl::and_< CR,
                        typename boost::mpl::contains< typename Agg::sorted_placeholders_t, Plc >::type >::type;
                };
            };

            template < typename CurrentResult, typename ESF >
            struct apply {
                using type = typename boost::mpl::fold< typename ESF::args_t,
                    CurrentResult,
                    typename investigate_placeholder< Aggregator >::template apply< boost::mpl::_1,
                                                            boost::mpl::_2 > >::type;
            };

            template < typename CurrentResult, typename ESFs >
            struct apply< CurrentResult, independent_esf< ESFs > > {
                using type =
                    typename boost::mpl::fold< ESFs, CurrentResult, apply< boost::mpl::_1, boost::mpl::_2 > >::type;
            };
        };

        template < typename CurrentResult, typename Aggregator, typename... RestOfMss >
        struct unwrap_esf_sequence;

        // Recursion base
        template < typename CurrentResult, typename Aggregator >
        struct unwrap_esf_sequence< CurrentResult, Aggregator > {
            using type = CurrentResult;
        };

        template < typename CurrentResult, typename Aggregator, typename FirstMss, typename... RestOfMss >
        struct unwrap_esf_sequence< CurrentResult, Aggregator, FirstMss, RestOfMss... > {
            using esfs = typename FirstMss::esf_sequence_t;
            using CR = typename boost::mpl::fold< esfs,
                CurrentResult,
                typename investigate_esf< Aggregator >::template apply< boost::mpl::_1, boost::mpl::_2 > >::type;
            using type = typename unwrap_esf_sequence< CR, Aggregator, RestOfMss... >::type;
        };

        template < typename CurrentResult,
            typename Aggregator,
            typename Mss0,
            typename Mss1,
            typename Tag,
            typename... RestOfMss >
        struct unwrap_esf_sequence< CurrentResult, Aggregator, condition< Mss0, Mss1, Tag >, RestOfMss... > {
            using CR1 = typename unwrap_esf_sequence< CurrentResult, Aggregator, Mss0 >::type;
            using CR2 = typename unwrap_esf_sequence< CR1, Aggregator, Mss1 >::type;
            using type = typename unwrap_esf_sequence< CR2, Aggregator, RestOfMss... >::type;
        };

        /**
           This metafuction is for debugging purpose. It checks that
           all the placeholders used in the making of a computation
           are also listed in the aggregator.
        */
        template < typename Aggregator, typename... Mss >
        struct all_args_in_aggregator {
            using type = typename unwrap_esf_sequence< boost::mpl::true_, Aggregator, Mss... >::type;
        };
    } // namespace _impl
} // namespace gridtools
