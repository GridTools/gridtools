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

#include <boost/mpl/and.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/contains.hpp>

namespace gridtools {

    namespace _impl {
        template <typename Aggregator>
        struct investigate_esf {
            template <typename Agg>
            struct investigate_placeholder {
                template <typename CR, typename Plc>
                struct apply {
                    using type = typename boost::mpl::and_<CR, typename boost::mpl::contains<typename Agg::placeholders, Plc>::type>::type;
                };
            };

            template <typename CurrentResult, typename ESF>
            struct apply {
                using type = typename boost::mpl::fold<
                    typename ESF::args_t,
                    CurrentResult,
                    typename investigate_placeholder<Aggregator>::template apply<boost::mpl::_1, boost::mpl::_2>
                    >::type;
            };

            template <typename CurrentResult, typename ESFs>
            struct apply<CurrentResult, independent_esf<ESFs> > {
                using type = typename boost::mpl::fold<
                    ESFs,
                    CurrentResult,
                    apply<boost::mpl::_1, boost::mpl::_2>
                    >::type;
            };
        };

        template <typename Aggregator, typename... RestOfMss>
        struct unwrap_esf_sequence;

        template <typename CurrentResult, typename Aggregator, typename FirstMss, typename... RestOfMss>
        struct unwrap_esf_sequence<CurrentResult, Aggregator, FirstMss, RestOfMss...> {
            using esfs = typename FirstMss::esf_sequence_t;
            using type = typename boost::mpl::fold<
                esfs,
                CurrentResult,
                typename investigate_esf<Aggregator>::template apply<boost::mpl::_1, boost::mpl::_2>
                >::type;
        };

        template <typename CurrentResult, typename Aggregator,
                  typename FirstMss0, typename FirstMss1, typename Tag, typename... RestOfMss>
        struct unwrap_esf_sequence<CurrentResult, Aggregator, condition<FirstMss0, FirstMss1, Tag>, RestOfMss...> {
            using tmp = typename unwrap_esf_sequence<CurrentResult, Aggregator, FirstMss0, RestOfMss...>::type;
            using type = typename unwrap_esf_sequence<tmp, Aggregator, FirstMss1, RestOfMss...>::type;
        };

        // SHORT CIRCUITING THE AND
        template <typename Aggregator, typename... RestOfMss>
        struct unwrap_esf_sequence<boost::mpl::false_, Aggregator, RestOfMss...> {
            using type = boost::mpl::false_;
        };

        // Recursion base
        template <typename Aggregator>
        struct unwrap_esf_sequence<Aggregator> {
            using type = boost::mpl::true_;
        };

        /**
           This metafuction is for debugging purpose. It checks that
           all the pplaceholders used in the making of a computation
           are also listed in the aggregator.
        */
        template <typename Aggregator, typename... Mss>
        struct all_args_in_aggregator {
            using type = typename unwrap_esf_sequence<boost::mpl::true_, Aggregator, Mss...>::type;
        }; // struct all_args_in_domain
    } // namespace _impl
} // namespace gridtools
