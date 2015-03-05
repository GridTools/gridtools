#pragma once

#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/reverse.hpp>
#include "level.h"

#include "../common/meta_array.h"
#include "axis.h"
#include "backend_traits_cuda.h"
#include "backend_traits_host.h"
#include "execution_types.h"
#include "mss_metafunctions.h"
#include "mss_local_domain.h"
#include "mss.h"
#include "strategy.h"
#include "run_functor_arguments.h"

/**
@file

\brief This file contains the traits which are used in backend.h
*/

namespace gridtools{
    /** enum defining the strategy policy for distributing the work. */

//    namespace _impl{

//forward declaration
    template<typename T>
    struct run_functor;

/**
@brief traits struct, specialized for the specific backends.
*/
    template<enumtype::backend Id>
    struct backend_traits_from_id{};

/** @brief functor struct whose specializations are responsible of running the kernel
    The kernel contains the computational intensive loops on the backend. The fact that it is a functor (and not a templated method) allows for partial specialization (e.g. two backends may share the same strategy)
*/
    template< typename Backend >
    struct execute_kernel_functor
    {
        template< typename Traits >
        static void execute_kernel( const typename Traits::local_domain_t& local_domain, const Backend * f);
    };

/**
   @brief traits struct for the run_functor

   empty declaration
*/
    template <class Subclass>
    struct run_functor_traits{};

/**
   @brief traits struct for the run_functor
   Specialization for all backend classes.
   This struct defines a type for all the template arguments in the run_functor subclasses. It is required because in the run_functor class definition the 'Derived'
   template argument is an incomplete type (ans thus we can not access its template arguments).
   This struct also contains all the type definitions common to all backends.
 */
    template <
        typename Arguments,
        template < typename Argument > class Back
    >
    struct run_functor_traits< Back< Arguments > >
    {
        BOOST_STATIC_ASSERT((is_run_functor_arguments<Arguments>::value));
        typedef Arguments arguments_t;
        typedef typename Arguments::local_domain_list_t local_domain_list_t;
        typedef typename Arguments::coords_t coords_t;
        typedef typename Arguments::functor_list_t functor_list_t;
        typedef typename Arguments::loop_intervals_t loop_intervals_t;
        typedef typename Arguments::functors_map_t functors_map_t;
        typedef typename Arguments::range_sizes_t range_sizes_t;
        typedef Back<Arguments> backend_t;

/**
   @brief traits class to be used inside the functor \ref gridtools::_impl::execute_kernel_functor, which dependson an Index type.
*/
            template <typename Index>
            struct traits{
                typedef typename boost::mpl::at<range_sizes_t, Index>::type range_t;
                typedef typename boost::mpl::at<functor_list_t, Index>::type functor_t;
                typedef typename boost::fusion::result_of::value_at<local_domain_list_t, Index>::type local_domain_t;
                typedef typename boost::mpl::at<functors_map_t, Index>::type interval_map_t;
                typedef typename index_to_level<
                    typename boost::mpl::deref<
                        typename boost::mpl::find_if<
                            loop_intervals_t,
                            boost::mpl::has_key<interval_map_t, boost::mpl::_1>
                            >::type
                        >::type::first
                    >::type first_hit_t;

                typedef typename local_domain_t::iterate_domain_t iterate_domain_t;
            };
    };

    template <enumtype::backend, uint_t Id>
    struct once_per_block{
    };
//    }//namespace _impl
}//namespace gridtools
