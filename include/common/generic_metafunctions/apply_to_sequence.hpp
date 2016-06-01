#pragma once
#include <boost/mpl/quote.hpp>
#include <boost/mpl/transform.hpp>

namespace gridtools {

    /*
     * @struct is_sequence_of
     * metafunction that determines if a mpl sequence is a sequence of types determined by the filter
     * @param TSeq sequence to query
     * @param TPred filter that determines the condition
     */
    template < typename Seq, template < typename > class Lambda >
    struct apply_to_sequence {
        typedef boost::mpl::quote1< Lambda > lambda_t;

        typedef typename boost::mpl::transform< Seq, lambda_t >::type type;
    };

} // namespace gridtools
