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
#include <boost/mpl/fold.hpp>
#include "../common/generic_metafunctions/is_sequence_of.hpp"

namespace gridtools {


    template <typename Mss1, typename Mss2, typename Cond>
    struct condition;

    template<typename Vector, typename ... Mss>
    struct meta_array_vector;

    template<typename Vector>
    struct meta_array_vector<Vector>{
        typedef Vector type;
    };

    template<typename Vector, typename First, typename ... Mss>
    struct meta_array_vector<Vector, First, Mss...>{
        typedef typename boost::mpl::push_front<typename meta_array_vector<Vector, Mss ...>::type , First>::type type;
    };

    template<typename Vector, typename Mss1, typename Mss2, typename Cond, typename ... Mss>
    struct meta_array_vector<Vector, condition<Mss1, Mss2, Cond>, Mss ... > {
        typedef condition<
            typename meta_array_vector<
                typename boost::mpl::push_front<Vector
                                                , Mss1>::type>::type
            , typename meta_array_vector<
                  typename  boost::mpl::push_front<Vector
                                                   , Mss2>::type>::type
            , Cond
            > type;
    };

/**
 * @brief wrapper class around a sequence of types. The goal of the class is to identify that a type is an array of types that
 * fulfil a predicate
 * (without having to inspect each element of the sequence)
 */
template<typename Sequence, typename TPred>
struct meta_array{
    typedef Sequence sequence_t;
    BOOST_STATIC_ASSERT((boost::mpl::is_sequence<sequence_t>::value));

    //check that predicate returns true for all elements
    typedef typename boost::mpl::fold<
        sequence_t,
        boost::mpl::true_,
        boost::mpl::and_<
            boost::mpl::_1,
            typename TPred::template apply<boost::mpl::_2>
        >
    >::type is_array_of_pred;

    BOOST_STATIC_ASSERT((is_array_of_pred::value));

    typedef sequence_t elements;
};

    //first order
    template<typename Sequence1, typename Sequence2, typename Cond, typename TPred>
    struct meta_array<condition<Sequence1, Sequence2, Cond>, TPred >{
        typedef Sequence1 sequence1_t;
        typedef Sequence2 sequence2_t;
        BOOST_STATIC_ASSERT((boost::mpl::is_sequence<sequence1_t>::value));
        BOOST_STATIC_ASSERT((boost::mpl::is_sequence<sequence2_t>::value));

        //check that predicate returns true for all elements
        typedef typename boost::mpl::fold<
            sequence1_t,
            boost::mpl::true_,
            boost::mpl::and_<
                boost::mpl::_1,
                typename TPred::template apply<boost::mpl::_2>
                >
            >::type is_array_of_pred1;

        // BOOST_STATIC_ASSERT((is_array_of_pred1::value));

        //check that predicate returns true for all elements
        typedef typename boost::mpl::fold<
            sequence2_t,
            boost::mpl::true_,
            boost::mpl::and_<
                boost::mpl::_1,
                typename TPred::template apply<boost::mpl::_2>
        >
            >::type is_array_of_pred2;

        // BOOST_STATIC_ASSERT((is_array_of_pred2::value));

        typedef condition<sequence1_t, sequence2_t, Cond> elements;
    };


//type traits for meta_array
template<typename T> struct is_meta_array : boost::mpl::false_{};

template<typename Sequence, typename TPred> struct is_meta_array< meta_array<Sequence, TPred> > : boost::mpl::true_{};

template<typename T, template<typename> class pred> struct is_meta_array_of : boost::mpl::false_{};

template<typename Sequence, typename pred, template<typename> class pred_query>
struct is_meta_array_of< meta_array<Sequence, pred>, pred_query>
{
    typedef typename boost::is_same<boost::mpl::quote1<pred_query>, pred >::type type;
    BOOST_STATIC_CONSTANT(bool, value=(type::value));
};

} //namespace gridtools
