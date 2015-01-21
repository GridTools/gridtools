#pragma once
#include "basic_token_execution.h"
#ifdef __CUDACC__
#include "cuda_profiler_api.h"
#endif

/**
@file Implementation of the k loop execution policy
The policies which are currently considered are
 - forward: the k loop is executed upward, increasing the value of the iterator on k. This is the option to be used when the stencil operations at level k depend on the fields at level k-1 (forward substitution).
 - backward: the k loop is executed downward, decreasing the value of the iterator on k. This is the option to be used when the stencil operations at level k depend on the fields at level k+1 (backward substitution).
 - parallel: the operations on each k level are executed in parallel. This is feasable only if there are no dependencies between levels.
*/

namespace gridtools{
    namespace _impl{

/**
   @brief   Execution kernel containing the loop over k levels
*/
        template< typename ExecutionEngine, typename ExtraArguments >
        struct run_f_on_interval{
            typedef uint_t local_domain_t;
        };

/**
   @brief partial specialization for the forward or backward cases
*/
        template<
            enumtype::execution IterationType,
            typename ExtraArguments>
        struct run_f_on_interval< typename enumtype::execute<IterationType>, ExtraArguments > : public run_f_on_interval_base< run_f_on_interval<typename enumtype::execute<IterationType>, ExtraArguments > >
        {
	    typedef run_f_on_interval_base< run_f_on_interval<typename enumtype::execute<IterationType>, ExtraArguments > > super;
            typedef typename enumtype::execute<IterationType>::type execution_engine;
            typedef ExtraArguments traits;


	    //////////////////////Compile time checks ////////////////////////////////////////////////////////////
	    //checking that all the placeholders have a different index
	    /**
	     * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the placehoolders
	     * note that the static const indexes are transformed into types using mpl::integral_c
	     */
	    typedef typename boost::mpl::transform<typename traits::functor_t::arg_list,
						   _impl::l_get_index
						   >::type raw_index_list;

	    static const uint_t len=boost::mpl::size<raw_index_list>::value;


	    //check if the indexes are repeated (a common error is to define 2 types with the same index)
	    //this method is the same one used in \ref gridtools::domain_type to verify that the indices are not repeated
	    typedef typename boost::mpl::fold<raw_index_list,
					      boost::mpl::set<>,
					      boost::mpl::insert<boost::mpl::_1, boost::mpl::_2>
					      >::type index_set;
	    //actual check if the user specified placeholder arguments with the same index
	    BOOST_STATIC_ASSERT((len == boost::mpl::size<index_set>::type::value ));

	    //checking if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)

	    typedef boost::mpl::range_c<uint_t ,0,len> range_t;
	    typedef typename boost::mpl::fold<range_t,
					      boost::mpl::vector<>,
					      boost::mpl::if_<boost::mpl::less<boost::mpl::at<raw_index_list, boost::mpl::_2>, static_int<len> >,
							      boost::mpl::push_back<
							      boost::mpl::_1,
								  boost::mpl::find<raw_index_list, boost::mpl::_2>
								  >,
							      boost::mpl::_1>
                                          >::type test_holes;
	    //actual check if the user specified placeholder arguments missing some indexes (there's a hole in the indexing)
	    BOOST_STATIC_ASSERT((len == boost::mpl::size<test_holes>::type::value ));
	    //////////////////////////////////////////////////////////////////////////////////////////////////////


            GT_FUNCTION
            explicit run_f_on_interval(typename traits::local_domain_t & domain, typename traits::coords_t const& coords):super(domain, coords){}


            template<typename IterationPolicy, typename IntervalType>
            GT_FUNCTION
            void loop(uint_t from, uint_t to) const {

	      for ( uint_t k=from ; k<=to; ++k, IterationPolicy::increment(this->m_domain)) {
	      	traits::functor_t::Do(this->m_domain, IntervalType());
		/* printf("k=%d\n", k); */
	      }

            }
        };

/**
   @brief partial specialization for the parallel case (to be implemented)
   stub
*/
        // template<
        //     typename ExtraArguments>
        // struct run_f_on_interval<typename enumtype::execute<enumtype::parallel>, ExtraArguments > : public run_f_on_interval_base< run_f_on_interval<typename enumtype::execute<enumtype::parallel>, ExtraArguments > >
        // {
	//     exit(-37);
        // };
    } // namespace _impl
} // namespace gridtools
