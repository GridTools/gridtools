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

    template < typename RunFunctorArguments, typename Pattern, typename Repl >
    struct transform_meta_data;

    template < template < typename... > class Metadata, typename... Args, typename Pattern, typename Repl >
    struct transform_meta_data< Metadata< Args... >, Pattern, Repl > {
        typedef Metadata< typename subs< Pattern, Repl, Args>::type... > type;
    };
}
