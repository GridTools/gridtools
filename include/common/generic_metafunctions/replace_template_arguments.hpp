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
