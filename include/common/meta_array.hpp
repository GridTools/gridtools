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
