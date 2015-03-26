#pragma once
#include <boost/mpl/has_key.hpp>
#include "iteration_policy.h"
#include "level.h"

namespace gridtools {
    namespace _impl {

        namespace{
            /**
               @brief generic forward declaration of the execution_policy struct.
            */
            template < typename RunF >
                struct execution_policy;

            /**
               @brief forward declaration of the execution_policy struct
            */
            template <
                typename Arguments, typename T,
                template < typename U, typename Argument > class Back
                >
                struct execution_policy<Back<T,Arguments> >
            {
                typedef Arguments traits_t;
                typedef T execution_engine_t;
            };
        }//unnamed namespace

/**
   @brief basic token of execution responsible of handling the discretization over the vertical dimension. This may be done with a loop over k or by partitoning the k axis and executing in parallel, depending on the execution_policy defined in the multi-stage stencil. The base class is then specialized using the CRTP pattern for the different policies.
*/
        template < typename Derived >
        struct run_f_on_interval_base {

            /**\brief necessary because the Derived class is an incomplete type at the moment of the instantiation of the base class*/
            typedef typename execution_policy<Derived>::traits_t traits;
            typedef typename execution_policy<Derived>::execution_engine_t execution_engine;

            GT_FUNCTION
            explicit run_f_on_interval_base(typename traits::local_domain_t & domain, typename traits::coords_t const& coords)
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
		    level_to_index<typename traits::coords_t::axis_type::FromLevel>::value <= Interval::first::value &&
		    level_to_index<typename traits::coords_t::axis_type::ToLevel>::value >= Interval::second::value , "the k interval exceeds the axis you specified for the coordinates instance")

		typedef iteration_policy<from_t, to_t, execution_engine::type::iteration> iteration_policy;

		if (boost::mpl::has_key<typename traits::interval_map_t, Interval>::type::value) {
		  typedef typename boost::mpl::at<typename traits::interval_map_t, Interval>::type interval_type;

                  uint_t from=m_coords.template value_at<from_t>();
		  //m_coords.template value_at<typename iteration_policy::from>();
                    uint_t to=m_coords.template value_at<to_t>();
                    /* uint_t to=m_coords.template value_at<typename iteration_policy::to>(); */
                    // std::cout<<"from==> "<<from<<std::endl;
                    // std::cout<<"to==> "<<to<<std::endl;
                    static_cast<Derived*>(const_cast<run_f_on_interval_base<Derived>* >(this))->template do_loop<iteration_policy, interval_type>(from, to);
                }

            }
        protected:
            typename traits::coords_t const &m_coords;
            typename traits::local_domain_t &m_domain;
        };

    } // namespace _impl
} // namespace gridtools
