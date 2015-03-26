/*
 * loop_interval.h
 *
 *  Created on: Feb 17, 2015
 *      Author: carlosos
 */

#pragma once
#include "functor_do_methods.h"
#include "loopintervals.h"
#include "functor_do_method_lookup_maps.h"

namespace gridtools {

template<
    typename functors_list,
    typename Coords
>
struct mss_intervals
{
    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename boost::mpl::transform<
        functors_list,
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

template<
    typename MssType,
    typename Coords
>
struct mss_functor_do_methods
{
    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename boost::mpl::transform<
        typename MssType::functors_list,
        compute_functor_do_methods<boost::mpl::_, typename Coords::axis_type>
    >::type type; // Vector of vectors - each element is a vector of pairs of actual axis-indices
};

template<
    typename MssType,
    typename Coords
>
struct mss_loop_intervals
{
    /**
     *  compute the functor do methods - This is the most computationally intensive part
     */
    typedef typename mss_functor_do_methods<MssType, Coords>::type functor_do_methods;

    /**
     * compute the loop intervals
     */
    typedef typename compute_loop_intervals<
        functor_do_methods,
        typename Coords::axis_type
    >::type type; // vector of pairs of indices - sorted and contiguous
};

template<
    typename MssType,
    typename Coords
>
struct mss_functor_do_method_lookup_maps
{
    typedef typename mss_functor_do_methods<MssType, Coords>::type functor_do_methods;

    typedef typename mss_loop_intervals<MssType, Coords>::type loop_intervals;
    /**
     * compute the do method lookup maps
     *
     */
    typedef typename boost::mpl::transform<
        functor_do_methods,
        compute_functor_do_method_lookup_map<boost::mpl::_, loop_intervals>
    >::type type; // vector of maps, indexed by functors indices in Functor vector.
};

} //namespace gridtools
