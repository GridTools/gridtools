/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
