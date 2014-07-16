#pragma once

#include <boost/mpl/has_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/find_if.hpp>
#include "../storage/storage.h"
#include "basic_token_execution.h"
#include "heap_allocated_temps.h"

#include "backend.h"

namespace gridtools {
    namespace _impl{
	template <>
	struct cout<_impl::OpenMP>
	{
	    template <typename T>
	    const cout& operator << (T arg) const {std::cout<<arg; return *this;}
	};

    }//namespace _impl

    namespace _impl_naive {


        template <typename FunctorList,
                  typename LoopIntervals,
                  typename FunctorsMap,
                  typename RangeSizes,
                  typename DomainList,
                  typename Coords>
	struct run_functor_naive : public _impl::run_functor < run_functor_naive<FunctorList, LoopIntervals, FunctorsMap, RangeSizes , DomainList, Coords > >
	{

	    typedef FunctorList functor_list_t;
	    typedef LoopIntervals loop_intervals_t;
	    typedef FunctorsMap functors_map_t;
	    typedef RangeSizes range_sizes_t;
	    typedef DomainList domain_list_t;
	    typedef Coords coords_t;

	    // useful if we can use constexpr
	    // static const _impl::BACKEND m_backend=_impl::OpenMP;
	    // static const _impl::BACKEND backend() {return m_backend;} //constexpr

            Coords const &coords;
            DomainList &domain_list;

            explicit run_functor_naive(DomainList & domain_list, Coords const& coords)
                : coords(coords)
                , domain_list(domain_list)
		{}

	    template< typename Traits >
	    struct execute_kernel_functor
	    {
		typedef run_functor_naive<FunctorList, LoopIntervals, FunctorsMap, RangeSizes, DomainList, Coords> backend_t;

		// template<_impl::STRATEGY s>
		static void wtf(){}

		template<_impl::STRATEGY s>
		static void execute_kernel( const typename Traits::local_domain_type& local_domain, const backend_t* f)
		{
		    typedef typename Traits::range_type range_type;
		    typedef typename Traits::functor_type functor_type;
		    typedef typename Traits::local_domain_type  local_domain_type;
		    typedef typename Traits::interval_map interval_map;
		    typedef typename Traits::iterate_domain_type iterate_domain_type;

		    for (int i = f->coords.i_low_bound() + range_type::iminus::value;
		    	 i < f->coords.i_high_bound() + range_type::iplus::value;
		    	 ++i)
		    	for (int j = f->coords.j_low_bound() + range_type::jminus::value;
		    	     j < f->coords.j_high_bound() + range_type::jplus::value;
		    	     ++j) {
		    	    iterate_domain_type it_domain(local_domain, i,j, f->coords.template value_at<typename Traits::first_hit>());

		    	    gridtools::for_each<LoopIntervals>
		    		(_impl::run_f_on_interval
		    		 <functor_type,
		    		 interval_map,
		    		 iterate_domain_type,
		    		 Coords>
		    		 (it_domain,f->coords)
		    		    );
		    	}
		}

	    };

	};
    }

    struct backend_naive: public heap_allocated_temps<backend_naive> {
        static const int BI = 0;
        static const int BJ = 0;
        static const int BK = 0;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef storage<ValueType, Layout> type;
        };

        template <typename ValueType, typename Layout>
        struct temporary_storage_type {
            typedef temporary<storage<ValueType, Layout> > type;
        };

        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename Domain, // Domain class (not really useful maybe)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList> // List of local domain to be pbassed to functor at<i>
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<FunctorList>::type::value> iter_range;

            gridtools::for_each<iter_range>(_impl::run_functor<_impl_naive::run_functor_naive
					    <
					    FunctorList,
					    LoopIntervals,
					    FunctorsMap,
					    range_sizes,
					    LocalDomainList,
					    Coords
					    >
					    >
					    (local_domain_list,coords));
        }
    };


    namespace _impl{

//wasted code because of the lack of constexpr
	template <typename FunctorList,
		  typename LoopIntervals,
		  typename FunctorsMap,
		  typename RangeSizes,
		  typename DomainList,
		  typename Coords>
	struct backend_type< _impl_naive::run_functor_naive<FunctorList, LoopIntervals, FunctorsMap, RangeSizes , DomainList, Coords> >
	{
	    static const BACKEND m_backend=OpenMP;
	};
    }

} // namespace gridtools
