#pragma once

/**
   @file

*/

namespace gridtools {
    namespace _impl {

	enum STRATEGY  {Naive, Block};
	enum BACKEND  {Cuda, OpenMP};


	template <BACKEND Backend>
	    struct cout{
		template <typename T>
		void operator <<(T t);
		//void endl();
	    };


//wasted code because of the lack of constexpr
	template <class RunFunctor>
	struct backend_type
	{};


/**
@brief traits struct for the run_functor

This struct defines a type for all the template arguments in the run_functor subclasses. It is required because in the run_functor class definition the 'Derived'
 template argument is an incomplete type (ans thus we can not access its template arguments).
This struct also contains all the type definitions common to all backends.
*/
template <class Subclass>
    struct run_functor_traits{};

template <
    typename FunctorList,
    typename LoopIntervals,
    typename FunctorsMap,
    typename RangeSizes,
    typename DomainList,
    typename Coords,
    template <typename FunctorList,typename  LoopIntervals,typename  FunctorsMap,typename  RangeSizes ,typename  DomainList,typename  Coords> class Back
>
    struct run_functor_traits< Back<FunctorList, LoopIntervals, FunctorsMap, RangeSizes , DomainList, Coords> >
{

    typedef FunctorList functor_list_t;
    typedef LoopIntervals loop_intervals_t;
    typedef FunctorsMap functors_map_t;
    typedef RangeSizes range_sizes_t;
    typedef DomainList domain_list_t;
    typedef Coords coords_t;
    typedef Back<FunctorList, LoopIntervals, FunctorsMap, RangeSizes , DomainList, Coords> type;

    template <typename Index>
    struct traits{
	typedef typename boost::mpl::at<range_sizes_t, Index>::type range_type;
	typedef typename boost::mpl::at<functor_list_t, Index>::type functor_type;
	typedef typename boost::fusion::result_of::value_at<domain_list_t, Index>::type local_domain_type;
	typedef typename boost::mpl::at<functors_map_t, Index>::type interval_map;
	typedef typename index_to_level<
	    typename boost::mpl::deref<
		typename boost::mpl::find_if<
		    loop_intervals_t,
		    boost::mpl::has_key<interval_map, boost::mpl::_1>
		    >::type
		>::type::first
	    >::type first_hit;

	typedef typename local_domain_type::iterate_domain_type iterate_domain_type;

    };



//    static const BACKEND m_backend = Back<FunctorList,LoopIntervals,FunctorsMap,RangeSizes ,DomainList,Coords>::m_backend;
};



/**
   \brief "base" struct for all the backend
   This class implements static polimorphism by means of the CRTP pattern. It contains all what is common for all the backends.
 */
        template < typename Derived >
	    struct run_functor {

	    typedef Derived derived_t;
	    typedef run_functor_traits<Derived> derived_traits;
	    typename derived_traits::coords_t const &coords;
	    typename derived_traits::domain_list_t &domain_list;

	    // this would be ok when using constexpr:
	    //static const BACKEND m_backend = derived_t::backend();

	    explicit run_functor(typename derived_traits::domain_list_t& domain_list, typename derived_traits::coords_t const& coords)
		: coords(coords)
		, domain_list(domain_list)
		{}

	    template <typename Index>
	    void operator()(Index const&) const {

#ifndef NDEBUG
		static const BACKEND backend_t = backend_type<derived_t>::m_backend;

//\todo a generic cout is still on the way (have to implement all the '<<' operators)
		cout< backend_t >() << "Functor " << /* functor_type() <<*/ "\n";
		/* cout<derived_traits::m_backend>() << "I loop " << coords.i_low_bound() + range_type::iminus::value << " -> " */
		/* 				      << (coords.i_high_bound() + range_type::iplus::value) << "\n"; */
		/* cout<derived_traits::m_backend>() << "J loop " << coords.j_low_bound() + range_type::jminus::value << " -> " */
		/* 		  << coords.j_high_bound() + range_type::jplus::value << "\n"; */
		cout< backend_t >() <<  " ******************** " /*<< first_hit()*/ << "\n";
		cout< backend_t >() << " ******************** " /*<< coords.template value_at<first_hit>()*/ << "\n";
#endif

		typename derived_traits::template traits<Index>::local_domain_type& local_domain = boost::fusion::at<Index>(domain_list);

        /////////////////////////// splitting in 2 steps (using non static method) //////////////////////////////
		// typename derived_traits::type::template execute_kernel_functor< typename derived_traits::template traits<Index> > functor;// temporary, possibly unnecessary
		// functor.template execute_kernel<_impl::Naive>(local_domain, static_cast<const derived_t*>(this));
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		derived_traits::type::template execute_kernel_functor< typename derived_traits::template traits<Index> >::template execute_kernel<_impl::Naive>(local_domain, static_cast<const derived_t*>(this));

	    }
	};

    }//namespace _impl
}//namespace gridtools
