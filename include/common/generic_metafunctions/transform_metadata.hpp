#pragma once

namespace gridtools {
    template < typename Pattern, typename Repl, typename Arg>
    struct subs {
        typedef typename boost::mpl::if_<
            boost::is_same<Pattern, Arg>,
            Repl,
            Arg
        >::type type;
    };

    /**
     * metafunction used to replace types stored in a metadata class.
     * When a type (Metadata) is used as "metadata" to store a collection of types,
     * transform_meta_data will substitute any type stored that matches a pattern (Pattern)
     * with a new value type (Repl).
     *
     * Usage example:
     * boost::is_same<
     *     typename transform_meta_data<wrap<int>, int, double>::type,
     *     wrap<double>
     * > :: value == true
     */
    template < typename Metadata, typename Pattern, typename Repl >
    struct transform_meta_data;

    template < template < typename... > class Metadata, typename... Args, typename Pattern, typename Repl >
    struct transform_meta_data< Metadata< Args... >, Pattern, Repl > {
        typedef Metadata< typename subs< Pattern, Repl, Args>::type... > type;
    };
}
