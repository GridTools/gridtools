/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include <boost/mpl/at.hpp>
#ifdef STRUCTURED_GRIDS
#include "expandable_parameters/iterate_domain_expandable_parameters.hpp"
#else
#include "icosahedral_grids/iterate_domain_expandable_parameters.hpp"
#endif
#include "run_functor_arguments.hpp"
#include "functor_decorator.hpp"

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

        template < ushort_t ID, typename Functor, typename IterateDomain, typename Interval >
        struct call_repeated {
          public:
            GT_FUNCTION
            static void call_do_method(IterateDomain &it_domain_) {

                typedef
                    typename boost::mpl::if_< typename boost::is_same< Interval,
                                                  typename Functor::f_with_default_interval::default_interval >::type,
                        typename boost::mpl::if_< typename has_do< typename Functor::f_type, Interval >::type,
                                                  typename Functor::f_type,
                                                  typename Functor::f_with_default_interval >::type,
                        typename Functor::f_type >::type functor_t;

                functor_t::Do(*static_cast< iterate_domain_expandable_parameters< IterateDomain, ID > * >(&it_domain_),
                    Interval());

                call_repeated< ID - 1, Functor, IterateDomain, Interval >::call_do_method(it_domain_);
            }
        };

        template < typename Functor, typename IterateDomain, typename Interval >
        struct call_repeated< 0, Functor, IterateDomain, Interval > {
          public:
            GT_FUNCTION
            static void call_do_method(IterateDomain &it_domain_) {}
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
	            also referenced in the functor. You get this error if you specify twice the same placeholder");
#endif

                static_cast< const RunEsfFunctorImpl * >(this)->template do_impl< interval_type, esf_arguments_t >();
            }
        }

      protected:
        iterate_domain_t &m_iterate_domain;
    };
} // namespace gridtools
