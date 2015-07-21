/*
 * loop_interval.h
 *
 *  Created on: Feb 17, 2015
 *      Author: carlosos
 */

#pragma once
#include "functor_do_methods.hpp"
#include "loopintervals.hpp"
#include "functor_do_method_lookup_maps.hpp"

namespace gridtools {

template<
    typename FunctorsList,
    typename Coords
>
struct mss_intervals
{
    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename boost::mpl::transform<
        FunctorsList,
        compute_functor_do_methods<boost::mpl::_, typename Coords::axis_type>
    >::type functor_do_methods; // Vector of vectors - each element is a vector of pairs of actual axis-indices

    /**
     * compute the loop intervals
     */
    typedef typename compute_loop_intervals<
        functor_do_methods,
        typename Coords::axis_type
    >::type loop_intervals_t; // vector of pairs of indices - sorted and contiguous

    /**
     * compute the do method lookup maps
     *
     */
    typedef typename boost::mpl::transform<
        functor_do_methods,
        compute_functor_do_method_lookup_map<boost::mpl::_, loop_intervals_t>
    >::type functor_do_method_lookup_maps; // vector of maps, indexed by functors indices in Functor vector.
};

} //namespace gridtools
