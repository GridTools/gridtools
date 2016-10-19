#pragma once
namespace gridtools {

#ifdef CXX11_ENABLED

    template < short_t Pre,
        typename InSequence,
        template < short_t... Args > class Sequence,
        short_t First,
        short_t... Args >
    struct recursive_expansion {
        typedef typename recursive_expansion< Pre, InSequence, Sequence, First - 1, First, Args... >::type type;
    };

    template < short_t Pre, typename InSequence, template < short_t... Args > class Sequence, short_t... Args >
    struct recursive_expansion< Pre, InSequence, Sequence, Pre, Args... > {
        typedef Sequence< boost::mpl::at_c< InSequence, Pre >::value, boost::mpl::at_c< InSequence, Args >::value... >
            type;
    };

    template < typename InSequence, template < short_t... Args > class Sequence, short_t Pre, short_t Post >
    struct gt_expand {
        typedef typename recursive_expansion< Pre, InSequence, Sequence, Post >::type type;
    };

#endif
} // namespace gridtools
