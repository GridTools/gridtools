#pragma once

#include <boost/mpl/quote.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/is_sequence.hpp>


/*
 * @struct is_sequence_of
 * metafunction that determines if a mpl sequence is a sequence of types determined by the filter
 * @param TSeq sequence to query
 * @param TPred filter that determines the condition
 */
template<typename TSeq, template<typename> class TPred>
struct is_sequence_of
{
    typedef boost::mpl::quote1<TPred> pred_t;

    typedef typename boost::mpl::lambda<
        boost::mpl::not_<
            typename pred_t::template apply<boost::mpl::_1>
        >
    >::type NegPred;

    typedef typename boost::mpl::eval_if<
        boost::mpl::is_sequence<TSeq>,
        boost::mpl::eval_if<
            boost::is_same<
                typename boost::mpl::find_if<TSeq, NegPred >::type,
                typename boost::mpl::end<TSeq>::type
            >,
            boost::mpl::true_,
            boost::mpl::false_
        >,
        boost::mpl::false_
    >::type type;

    BOOST_STATIC_CONSTANT(bool, value = (type::value) );
};

