#pragma once
namespace gridtools {

#ifdef CXX11_ENABLED

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
                boost::mpl::at_c< InSequence, Args >::type::value... >
                type;
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

#endif
} // namespace gridtools
