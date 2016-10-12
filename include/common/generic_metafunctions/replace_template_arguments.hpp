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

namespace gridtools {
#if defined(CXX11_ENABLED) && !defined(__CUDACC__)
    template < typename Pattern, typename Repl, typename Arg >
    struct subs {
        typedef typename boost::mpl::if_< boost::is_same< Pattern, Arg >, Repl, Arg >::type type;
    };

    /**
     * metafunction used to replace types stored in a metadata class.
     * When a type (Metadata) is used as "metadata" to store a collection of types,
     * replace_template_arguments will substitute any type stored that equals a pattern (Pattern)
     * with a new value type (Repl).
     *
     * Usage example:
     * boost::is_same<
     *     typename replace_template_arguments<wrap<int>, int, double>::type,
     *     wrap<double>
     * > :: value == true
     */
    template < typename Metadata, typename Pattern, typename Repl >
    struct replace_template_arguments;

    template < template < typename... > class Metadata, typename... Args, typename Pattern, typename Repl >
    struct replace_template_arguments< Metadata< Args... >, Pattern, Repl > {
        typedef Metadata< typename subs< Pattern, Repl, Args >::type... > type;
    };
#endif
} // namespace gridtools
