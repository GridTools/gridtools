#pragma once
#include <boost/static_assert.hpp>
#include "basic_token_execution.h"
#include "backend_traits_fwd.h"
#include "run_esf_functor.h"

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
        template<
            typename ExecutionEngine,
            typename RunFunctorArguments
        >
        struct run_f_on_interval{
            typedef uint_t local_domain_t;
        };

        /**
           @brief partial specialization for the forward or backward cases
        */
        template<
            enumtype::execution IterationType,
            typename RunFunctorArguments
        >
        struct run_f_on_interval<
            enumtype::execute<IterationType>,
            RunFunctorArguments
        > : public run_f_on_interval_base< run_f_on_interval<
                    enumtype::execute<IterationType>, RunFunctorArguments
                >
            > //CRTP
        {
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));

            typedef typename backend_traits_from_id<RunFunctorArguments::backend_id_t::value >::run_esf_functor_h_t run_esf_functor_h_t;
            typedef run_f_on_interval_base<
                        run_f_on_interval<
                            typename enumtype::execute<IterationType>,
                            RunFunctorArguments
                        >
                    > super;
            typedef typename super::iterate_domain_t iterate_domain_t;
            typedef typename enumtype::execute<IterationType>::type execution_engine;
            typedef typename RunFunctorArguments::functor_list_t functor_list_t;


            //////////////////////Compile time checks ////////////////////////////////////////////////////////////
            //checking that all the placeholders have a different index
            /**
             * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the placehoolders
             * note that the static const indexes are transformed into types using mpl::integral_c
             */
//            typedef _impl::compute_index_set<typename functor_t::arg_list> check_holes;
//            typedef typename check_holes::raw_index_list raw_index_list;
//            typedef typename check_holes::index_set index_set;
//            static const ushort_t len=check_holes::len;
//
//            //actual check if the user specified placeholder arguments with the same index
//            GRIDTOOLS_STATIC_ASSERT((len == boost::mpl::size<index_set>::type::value ), "You specified different placeholders with the same index. Check the indexes of the arg_type definitions.")
//
//            //checking if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
//            typedef typename boost::mpl::find_if<raw_index_list, boost::mpl::greater<boost::mpl::_1, static_int<len-1> > >::type test;
//            //check if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
//            GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename test::type, boost::mpl::void_ >::value) , "the index list contains holes:\n\
//The numeration of the placeholders is not contiguous. You have to define each arg_type with a unique identifier ranging from 1 to N without \"holes\".")
//            //////////////////////////////////////////////////////////////////////////////////////////////////////


            GT_FUNCTION
            explicit run_f_on_interval(
                    iterate_domain_t & iterate_domain,
                    typename RunFunctorArguments::coords_t const& coords):
                super(iterate_domain, coords){}

            template<typename IterationPolicy, typename Interval>
            GT_FUNCTION
            void loop(uint_t from, uint_t to) const {
                typedef typename run_esf_functor_h_t::template apply<RunFunctorArguments, Interval>::type run_esf_functor_t;

                for ( uint_t k=from ; k<=to; ++k, IterationPolicy::increment(this->m_domain)) {
                    gridtools::for_each<boost::mpl::range_c<int, 0, boost::mpl::size<functor_list_t>::value > > (
                        run_esf_functor_t(this->m_domain)
                    );
                }
            }

        };

        /**
           @brief partial specialization for the parallel case (to be implemented)
           stub
         */
        template<
            typename RunFunctorArguments
        >
        struct run_f_on_interval<
            typename enumtype::execute<enumtype::parallel>,
            RunFunctorArguments
        > : public run_f_on_interval_base<
            run_f_on_interval<
                enumtype::execute<enumtype::parallel>,
                RunFunctorArguments
            >
        > {
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));
            //*TODO implement me
        };
    } // namespace _impl
} // namespace gridtools
