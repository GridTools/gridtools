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
namespace gridtools {

    namespace impl_ {
        template < short_t Pre,
            typename InSequence,
            template < short_t... Args > class Sequence,
            short_t First,
            short_t... Args >
        struct recursive_expansion {
            typedef typename recursive_expansion< Pre, InSequence, Sequence, First - 1, First, Args... >::type type;
        };

        template < short_t Pre, typename InSequence, template < short_t... > class Sequence, short_t... Args >
        struct recursive_expansion< Pre, InSequence, Sequence, Pre, Args... > {
            typedef Sequence< boost::mpl::at_c< InSequence, Pre >::type::value,
                boost::mpl::at_c< InSequence, Args >::type::value... > type;
        };
    }

    /**
       @brief metafunction thet given a contaner with integer template argument and
       an boost::mpl::vector_c representing its arugments returns the
       container with a subset of the arguments

       \tparam InSequence input boost::mpl sequence (must work with boost::mpl::at_c)
       \tparam Sequence container to be filled with the subset of indices
       \tparam Pre position of the fist index for the subsequence
       \tparam Post position of the last index for the subsequence

       usage with \ref gridtools::layout_map, int the sub_map metafunction
     */
    template < typename InSequence, template < short_t... Args > class Sequence, short_t Pre, short_t... Post >
    struct gt_expand {
        typedef typename impl_::recursive_expansion< Pre, InSequence, Sequence, Post... >::type type;
    };

} // namespace gridtools
