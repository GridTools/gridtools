#pragma once

/**
   @file

*/

namespace gridtools {
    namespace _impl {

        enum STRATEGY  {Naive, Block};
        enum BACKEND  {Cuda, Host};


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


            /**
             * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
             */
            template <typename Index>
            void operator()(Index const&) const {

#ifndef NDEBUG
                static const BACKEND backend_t = backend_type<derived_t>::m_backend;

                typedef typename derived_traits::template traits<Index>::range_type range_type;
//\todo a generic cout is still on the way (have to implement all the '<<' operators)
                cout< backend_t >() << "Functor " <<  typename derived_traits::template traits<Index>::functor_type() << "\n";
                cout< backend_t >() << "I loop " << coords.i_low_bound() + range_type::iminus::value << " -> "
                                    << (coords.i_high_bound() + range_type::iplus::value) << "\n";
                cout< backend_t >() << "J loop " << coords.j_low_bound() + range_type::jminus::value << " -> "
                                    << coords.j_high_bound() + range_type::jplus::value << "\n";
                cout< backend_t >() <<  " ******************** " /*<< first_hit()*/ << "\n";
                cout< backend_t >() << " ******************** " /*<< coords.template value_at<first_hit>()*/ << "\n";
#endif

                typename derived_traits::template traits<Index>::local_domain_type& local_domain = boost::fusion::at<Index>(domain_list);

                /////////////////////////// splitting in 2 steps (using non static method) //////////////////////////////
                // typename derived_traits::type::template execute_kernel_functor< typename derived_traits::template traits<Index> > functor;// temporary, possibly unnecessary
                // functor.template execute_kernel<_impl::Naive>(local_domain, static_cast<const derived_t*>(this));
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                typedef typename derived_traits::type::template execute_kernel_functor< typename derived_traits::template traits<Index> > functor_type;
                functor_type::template execute_kernel<_impl::Naive>(local_domain, static_cast<const derived_t*>(this));

            }
        };

        template<BACKEND Id>
        struct backend_from_id
        {
        };

    }//namespace _impl


/**this struct contains the 'run' method for all backends, with a policy determining the specific type. Each backend contains a traits class for the specific case.*/
    template<_impl::BACKEND BackendType>
    struct backend: public heap_allocated_temps<backend<BackendType> > {
        static const int BI = 0;
        static const int BJ = 0;
        static const int BK = 0;

        typedef _impl::backend_from_id <BackendType> backend_traits;

        template <typename ValueType, typename Layout>
        struct storage_type {
            typedef typename backend_traits::template storage_traits<ValueType, Layout>::storage_type type;
        };

        template <typename ValueType, typename Layout>
        struct temporary_storage_type {
            typedef temporary< typename backend_traits::template storage_traits<ValueType, Layout>::storage_type > type;
        };


        /**
         * \brief calls the \ref gridtools::run_functor for each functor in the FunctorList.
         * the loop over the functors list is unrolled at compile-time using the for_each construct.
         * \tparam FunctorList  List of functors to execute (in order)
         * \tparam range_sizes computed range sizes to know where to compute functot at<i>
         * \tparam LoopIntervals List of intervals on which functors are defined
         * \tparam FunctorsMap Map between interval and actual arguments to pass to Do methods
         * \tparam Domain Domain class (not really useful maybe)
         * \tparam Coords Coordinate class with domain sizes and splitter coordinates
         * \tparam LocalDomainList List of local domain to be pbassed to functor at<i>
         */
        template <typename FunctorList, // List of functors to execute (in order)
                  typename range_sizes, // computed range sizes to know where to compute functot at<i>
                  typename LoopIntervals, // List of intervals on which functors are defined
                  typename FunctorsMap,  // Map between interval and actual arguments to pass to Do methods
                  typename Domain, // Domain class (not really useful maybe)
                  typename Coords, // Coordinate class with domain sizes and splitter coordinates
                  typename LocalDomainList> // List of local domain to be pbassed to functor at<i>
        static void run(Domain const& domain, Coords const& coords, LocalDomainList &local_domain_list) {

            typedef boost::mpl::range_c<int, 0, boost::mpl::size<FunctorList>::type::value> iter_range;

            backend_traits::template for_each<iter_range>(_impl::run_functor<typename backend_traits::template execute_traits
                                                          <
                                                          FunctorList,
                                                          LoopIntervals,
                                                          FunctorsMap,
                                                          range_sizes,
                                                          LocalDomainList,
                                                          Coords
                                                          >::run_functor
                                                          >
                                                          (local_domain_list,coords));
        }
    };



}//namespace gridtools
