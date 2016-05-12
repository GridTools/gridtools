#pragma once
#include <boost/mpl/has_key.hpp>
#include "iteration_policy.hpp"
#include "level.hpp"
#include "interval.hpp"

namespace gridtools {
    namespace _impl {

        namespace{
            /**
               @brief generic forward declaration of the execution_policy struct.
            */
            template < typename RunF >
            struct run_f_on_interval_esf_arguments;

            template < typename RunF >
            struct run_f_on_interval_run_functor_arguments;

            template < typename RunF >
            struct run_f_on_interval_execution_engine;

            /**
               @brief forward declaration of the execution_policy struct
            */
            template <
                typename ExecutionEngine, typename RunFunctorArguments,
                template<typename, typename> class Impl
            >
            struct run_f_on_interval_run_functor_arguments<Impl<ExecutionEngine, RunFunctorArguments> >
            {
                typedef RunFunctorArguments type;
            };
            template <
                typename ExecutionEngine, typename RunFunctorArguments,
                template<typename, typename> class Impl
            >
            struct run_f_on_interval_execution_engine<Impl<ExecutionEngine, RunFunctorArguments> >
            {
                typedef ExecutionEngine type;
            };

        }//unnamed namespace

/**
   @brief basic token of execution responsible of handling the discretization over the vertical dimension. This may be done with a loop over k or by partitoning the k axis and executing in parallel, depending on the execution_policy defined in the multi-stage stencil. The base class is then specialized using the CRTP pattern for the different policies.
*/
        template < typename RunFOnIntervalImpl >
        struct run_f_on_interval_base {
            /**\brief necessary because the Derived class is an incomplete type at the moment of the instantiation of the base class*/
            typedef typename run_f_on_interval_run_functor_arguments<RunFOnIntervalImpl>::type run_functor_arguments_t;
            typedef typename run_f_on_interval_execution_engine<RunFOnIntervalImpl>::type execution_engine;

            typedef typename run_functor_arguments_t::local_domain_t local_domain_t;
            typedef typename run_functor_arguments_t::iterate_domain_t iterate_domain_t;
            typedef typename run_functor_arguments_t::coords_t coords_t;

            GT_FUNCTION
            explicit run_f_on_interval_base(iterate_domain_t & domain, coords_t const& coords)
                : m_coords(coords)
                , m_domain(domain)
            {}

            template <typename Interval>
            GT_FUNCTION
            void operator()(Interval const&) const {
                typedef typename index_to_level<typename Interval::first>::type from_t;
                typedef typename index_to_level<typename Interval::second>::type to_t;

                //check that the axis specified by the user are containing the k interval
                GRIDTOOLS_STATIC_ASSERT(
                    (level_to_index<typename coords_t::axis_type::FromLevel>::value <= Interval::first::value &&
                    level_to_index<typename coords_t::axis_type::ToLevel>::value >= Interval::second::value) ,
                    "the k interval exceeds the axis you specified for the coordinates instance");


                typedef iteration_policy<from_t, to_t, execution_engine::type::iteration> iteration_policy;

                uint_t const from=m_coords.template value_at<from_t>();
                uint_t const to=m_coords.template value_at<to_t>();

                static_cast<RunFOnIntervalImpl*>(const_cast<run_f_on_interval_base<RunFOnIntervalImpl>* >(this))->
                        template k_loop<iteration_policy, Interval>(from, to);
            }

        protected:
            coords_t const &m_coords;
            iterate_domain_t &m_domain;
        };

    } // namespace _impl
} // namespace gridtools
