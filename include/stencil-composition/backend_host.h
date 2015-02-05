#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>

#include "execution_policy.h"
#include "heap_allocated_temps.h"
#include "backend.h"

#include "iteration_policy.h"

/**
   @file
   @brief Implements the stencil operations for the host backend
 */

namespace gridtools {

    namespace _impl_host {

/** @brief Host backend
    Derived class of the CRTP pattern defined in \ref gridtools::_impl::run_functor */
        template < typename Arguments >
        struct run_functor_host : public _impl::run_functor < run_functor_host< Arguments > >
        {

            typedef _impl::run_functor < run_functor_host < Arguments > > super;
            explicit run_functor_host(typename Arguments::domain_list_t& domain_list,  typename Arguments::coords_t const& coords)
                : super(domain_list, coords)
            {}

            explicit run_functor_host(typename Arguments::domain_list_t& domain_list,  typename Arguments::coords_t const& coords, uint_t i, uint_t j, uint_t bi, uint_t bj, uint_t blki, uint_t blkj)
                : super(domain_list, coords, i, j, bi, bj, blki, blkj)
            {}

        };
    }

    // namespace _impl{

            // define an SFINAE structure
            template <typename T>
            struct SFINAE;

            template <>
            struct SFINAE<int>{};


/** @brief Partial specialization: naive and block implementation for the host backend */
    template <typename Arguments >
    struct execute_kernel_functor < _impl_host::run_functor_host< Arguments > >
    {
        typedef _impl_host::run_functor_host< Arguments > backend_t;

        template <typename FunctorType, typename IntervalMapType, typename IterateDomainType, typename CoordsType>
        struct extra_arguments{
            typedef FunctorType functor_t;
            typedef IntervalMapType interval_map_t;
            typedef IterateDomainType local_domain_t;
            typedef CoordsType coords_t;};

        // template < typename RangeType, typename OtherRange >
        // struct extend_range{
        //     typedef typename sum_range<RangeType, OtherRange >::type type;
        // };


        template<typename TFunctor>
        struct sfinae
        {
            // define a MixIn class providing a Do member
            struct MixIn
            {
                using xrange = int ;
            };
            // multiple inheritance form TFunctor and MixIn
            // (if both define a Do it will be ambiguous in the Derived struct)
            struct derived : public TFunctor, public MixIn {};


            // SFINAE test methods which try to match the MixIn Do method signature
            // (note that multiple inheritance hides all symbols which are defined in more than one base class,
            // i.e. if TFunctor and MixIn both define a Do symbol then it will be ambiguous in the Derived class
            // and we will fall back to the ellipsis test method)
            template<typename TDerived>
            static boost::mpl::false_ test( SFINAE<typename TDerived::xrange>* x );
            template<typename TDerived>
            static boost::mpl::true_ test(...);

            typedef decltype(test<derived>(0)) type;
            typedef TFunctor functor_t;
        };


        template<typename Sfinae>
        struct extend_loop_bounds
        {
            // template <typename RangeType, typename Dummy=void>
            // struct apply{
            //     typedef typename sum_range<RangeType, decltype(Sfinae::functor_t::xrange)>::type type;
            // };

            template <typename RangeType, typename Dummy=void>
            struct apply;

            template <typename RangeType>
            struct apply<RangeType, typename boost::enable_if<typename Sfinae::type>::type >{
                typedef /*typename sum_range<*/RangeType/*, decltype(Sfinae::functor_t::xrange)>::type*/ type;
            };

            template <typename RangeType>
            struct apply<RangeType, typename boost::disable_if<typename Sfinae::type>::type >{
                typedef RangeType type;
            };

        };

        template<bool Cond, typename True, typename False, typename Dummy=void>
        struct static_if;

        template< typename True, typename False>
        struct static_if<true, True, False>{
            typedef True type;
        };

        template<typename True, typename False>
        struct static_if<false, True, False>{
            typedef False type;
        };

        template<typename Functor>
        struct get_xrange{
            struct apply{
                typedef typename Functor::xrange type;
            };
            typedef typename Functor::xrange type;
        };
/**
   @brief core of the kernel execution
   \tparam Traits traits class defined in \ref gridtools::_impl::run_functor_traits
*/
            template< typename Traits >
            static void execute_kernel( typename Traits::local_domain_t& local_domain, const backend_t * f )
                {
                    typedef typename Arguments::coords_t coords_type;
                    typedef typename Arguments::loop_intervals_t loop_intervals_t;
                    typedef typename Traits::range_t range_t;
                    typedef typename Traits::functor_t functor_type;
                    typedef typename Traits::local_domain_t  local_domain_t;
                    typedef typename Traits::interval_map_t interval_map_type;
                    typedef typename Traits::iterate_domain_t iterate_domain_type;
                    typedef typename Arguments::execution_type_t execution_type_t;

                    //typedef typename sfinae<functor_type>::type::fuck fuck;
                    typedef typename boost::mpl::eval_if_c<sfinae<functor_type>::type::value, get_xrange< functor_type >, boost::mpl::identity<range<0,0,0> > >::type new_range_t;

                    typedef typename sum_range<new_range_t, range_t>::type xrange_t;
                    //typedef typename extend_loop_bounds<sfinae<functor_type> >::template apply<range_t>::type xrange_t;
                    //typedef typename xrange_t::fuck fuck;

#ifndef NDEBUG
		    std::cout << "Functor " <<  functor_type() << "\n";
		    std::cout << "I loop " << (int_t)f->m_starti <<"+"<< xrange_t::iminus::value << " -> "
			      << f->m_starti <<"+"<< f->m_BI <<"+"<< xrange_t::iplus::value << "\n";
		    std::cout << "J loop " << (int_t)f->m_startj <<"+"<< xrange_t::jminus::value << " -> "
			      << (int_t)f->m_startj <<"+"<< f->m_BJ <<"+"<< xrange_t::jplus::value << "\n";
		    std::cout <<  " ******************** " << typename Traits::first_hit_t() << "\n";
		    std::cout << " ******************** " << f->m_coords.template value_at<typename Traits::first_hit_t>() << "\n";
		    std::cout<<"iminus::value: "<<xrange_t::iminus::value<<std::endl;
#endif

		    typename iterate_domain_type::float_t* data_pointer[Traits::iterate_domain_t::N_DATA_POINTERS];
		    iterate_domain_type it_domain(local_domain);
		    it_domain.template assign_storage_pointers<enumtype::Host>(data_pointer);

                    for (int_t i = (int_t)f->m_starti + xrange_t::iminus::value;
                         i <= (int_t)f->m_starti + (int_t)f->m_BI + xrange_t::iplus::value;
                         ++i)
                    {
			// for_each<local_domain.local_args>(increment<0>);
                        for (int_t j = (int_t)f->m_startj + xrange_t::jminus::value;
                             j <= (int_t)f->m_startj + (int_t)f->m_BJ + xrange_t::jplus::value;
                             ++j)
                        {
                            // for_each<local_domain.local_args>(increment<1>());
//#ifndef NDEBUG
                            //std::cout << "Move to : " << i << ", " << j << std::endl;
//#endif
                            //reset the index
                            it_domain.set_index(0);
                            it_domain.template assign_ij<0>(i, f->blk_idx_i);
                            it_domain.template assign_ij<1>(j, f->blk_idx_j);
                            /** setting an iterator to the address of the current i,j entry to be accessed */
                            typedef typename boost::mpl::front<loop_intervals_t>::type interval;
                            typedef typename index_to_level<typename interval::first>::type from;
                            typedef typename index_to_level<typename interval::second>::type to;
                            typedef _impl::iteration_policy<from, to, execution_type_t::type::iteration> iteration_policy;
                            assert(i>=0);
                            assert(j>=0);

                            //printf("setting the start to: %d \n",f->m_coords.template value_at< typename iteration_policy::from >() );
                            //setting the initial k level (for backward/parallel iterations it is not 0)
                            if( !(iteration_policy::value==enumtype::forward) )
                                it_domain.set_k_start( f->m_coords.template value_at< typename iteration_policy::from >() );

                            //local structs can be passed as template arguments in C++11 (would improve readability)
                            // struct extra_arguments{
                            //     typedef functor_type functor_t;
                            //     typedef interval_map_type interval_map_t;
                            //     typedef iterate_domain_type local_domain_t;
                            //     typedef coords_type coords_t;};

                            /** run the iteration on the k dimension */
                            gridtools::for_each< loop_intervals_t >
                                (_impl::run_f_on_interval
                                 <
                                 execution_type_t,
                                 extra_arguments<functor_type, interval_map_type, iterate_domain_type, coords_type>
				   >
				   (it_domain,f->m_coords)
				      );
                            }
		      }
                }

        };


/**
   @brief given the backend \ref gridtools::_impl_host::run_functor_host returns the backend ID gridtools::enumtype::Host
   wasted code because of the lack of constexpr
*/
        template <typename Arguments >
        struct backend_type< _impl_host::run_functor_host< Arguments > >
        {
            static const enumtype::backend s_backend=enumtype::Host;
        };

    // } //namespace _impl

} // namespace gridtools
