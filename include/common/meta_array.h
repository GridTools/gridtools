/*
 * meta_array.h
 *
 *  Created on: Feb 20, 2015
 *      Author: carlosos
 */

#pragma once
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/assert.hpp>

/*
 * @struct is_sequence_of
 * metafunction that determines if a mpl sequence is a sequence of types determined by the filter
 * @param TSeq sequence to query
 * @param TPred filter that determines the condition
 */
template<typename T> struct printj{BOOST_MPL_ASSERT_MSG((false), JJJJJJJJJJJJJJJ, (T));};
template<typename TSeq, typename TPred>
struct is_sequence_of
{
    typedef typename boost::mpl::lambda<
        boost::mpl::not_<
            typename TPred::template apply<boost::mpl::_1>
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

template<typename sequence>
struct meta_array{
    BOOST_STATIC_ASSERT((boost::mpl::is_sequence<sequence>::value));
    typedef typename boost::mpl::eval_if<
        boost::mpl::empty<sequence>,
        boost::mpl::identity<boost::mpl::void_>,
        boost::mpl::front<sequence>
    >::type elements;

    template<typename T> struct equal_to_elements {
        typedef typename boost::is_same<T, elements>::type type;
    };

    BOOST_MPL_ASSERT_MSG((false), TTTTT, (int));
    printj<elements> lpo;
    //TODO
    BOOST_STATIC_ASSERT((is_sequence_of<sequence, boost::mpl::quote1<equal_to_elements > >::value));
};

template<typename T> struct is_meta_array : boost::mpl::false_{};

template<typename sequence> struct is_meta_array< meta_array<sequence> > : boost::mpl::true_{};

template<typename T> struct meta_array_elements : boost::mpl::void_{};

template<typename sequence>
struct meta_array_elements<meta_array<sequence> >
{
    typedef typename meta_array<sequence>::elements type;
};

/*
 * @struct is_meta_array_of
 * metafunction that determines if a mpl sequence is a sequence of types determined by the filter
 * @param TSeq sequence to query
 * @param TPred filter that determines the condition
 */
template<typename TMetaArray, typename TPred>
struct is_meta_array_of
{
    typedef typename boost::mpl::and_<
        is_meta_array<TMetaArray>,
        typename TPred::template apply<typename meta_array_elements<TMetaArray>::type >
    >::type type;

    BOOST_STATIC_CONSTANT(bool, value = (type::value) );
};
