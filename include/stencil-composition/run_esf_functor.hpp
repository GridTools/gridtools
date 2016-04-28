#pragma once
#include <boost/mpl/at.hpp>
#include "backend.hpp"
#include "expandable_parameters/iterate_domain_expandable_parameters.hpp"

namespace gridtools {
    namespace _impl {
        template < typename RunEsfFunctorImpl >
        struct run_esf_functor_run_functor_arguments;

        template < typename RunEsfFunctorImpl >
        struct run_esf_functor_interval;

        template < typename RunFunctorArguments, typename Interval, template < typename, typename > class Impl >
        struct run_esf_functor_run_functor_arguments< Impl< RunFunctorArguments, Interval > > {
            typedef RunFunctorArguments type;
        };
        template < typename RunFunctorArguments, typename Interval, template < typename, typename > class Impl >
        struct run_esf_functor_interval< Impl< RunFunctorArguments, Interval > > {
            typedef Interval type;
        };

        template<ushort_t ID, typename Functor, typename IterateDomain, typename Interval>
        struct call_repeated{
        public:

            GT_FUNCTION
            static void Do(IterateDomain& it_domain_){

                Functor::f_type::Do(
                    *static_cast<iterate_domain_expandable_parameters<
                    IterateDomain
                    , ID> *
                    > (&it_domain_), Interval());

                call_repeated<ID-1, Functor, IterateDomain, Interval>::Do(it_domain_);
            }
        };

        template<typename Functor, typename IterateDomain, typename Interval>
        struct call_repeated<0, Functor, IterateDomain, Interval>{
        public:

            GT_FUNCTION
            static void Do(IterateDomain& it_domain_){
            }
        };

    }

    /**
       \brief "base" struct for all the backend
       This class implements static polimorphism by means of the CRTP pattern. It contains all what is common for all
       the backends.
    */
    template < typename RunEsfFunctorImpl >
    struct run_esf_functor {
        typedef
            typename _impl::run_esf_functor_run_functor_arguments< RunEsfFunctorImpl >::type run_functor_arguments_t;
        typedef typename _impl::run_esf_functor_interval< RunEsfFunctorImpl >::type interval_t;

        GRIDTOOLS_STATIC_ASSERT(
            (is_run_functor_arguments< run_functor_arguments_t >::value), "Internal Error: invalid type");
        typedef typename run_functor_arguments_t::iterate_domain_t iterate_domain_t;
        typedef typename run_functor_arguments_t::functor_list_t run_functor_list_t;

        GT_FUNCTION
        explicit run_esf_functor(iterate_domain_t &iterate_domain) : m_iterate_domain(iterate_domain) {}

        /**
         * \brief given the index of a functor in the functors
         * list, it calls a kernel on the GPU executing the
         * operations defined on that functor.
         */
        template < typename Index >
        GT_FUNCTION void operator()(Index const &) const {

            typedef esf_arguments< run_functor_arguments_t, Index > esf_arguments_t;

            typedef typename esf_arguments_t::interval_map_t interval_map_t;
            typedef typename esf_arguments_t::esf_args_map_t esf_args_map_t;
            typedef typename esf_arguments_t::functor_t functor_t;

            if (boost::mpl::has_key< interval_map_t, interval_t >::type::value) {
                typedef typename boost::mpl::at< interval_map_t, interval_t >::type interval_type;

// check that the number of placeholders passed to the elementary stencil function
//(constructed during the computation) is the same as the number of arguments referenced
// in the functor definition (in the high level interface). This means that we cannot
// (although in theory we could) pass placeholders to the computation which are not
// also referenced in the functor.

#ifdef PEDANTIC // we might want to use the same placeholder twice?
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::size< esf_args_map_t >::value ==
                        boost::mpl::size<
                            typename boost::mpl::at< run_functor_list_t, Index >::type::f_type::arg_list >::value),
                    "GRIDTOOLS ERROR:\n\
	            check that the number of placeholders passed to the elementary stencil function\n \
	            (constructed during the computation) is the same as the number of arguments referenced\n \
	            in the functor definition (in the high level interface). This means that we cannot\n \
	            (although in theory we could) pass placeholders to the computation which are not\n \
	            also referenced in the functor.");
#endif

                static_cast< const RunEsfFunctorImpl * >(this)->template do_impl< interval_type, esf_arguments_t >();
            }
        }

      protected:
        iterate_domain_t &m_iterate_domain;
    };
} // namespace gridtools
