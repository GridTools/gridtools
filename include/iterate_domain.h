#pragma once

#include <boost/fusion/view/zip_view.hpp>

namespace gridtools {

    namespace iterate_domain_aux {
        struct assign_iterators {
            int i, j, k;

            GT_FUNCTION
            assign_iterators(int i, int j, int k)
                : i(i)
                , j(j)
                , k(k)
            {}

            template <typename ZipElem>
            GT_FUNCTION
            void operator()(ZipElem const& ze) const {
                boost::fusion::at_c<0>(ze) = &( (*(boost::fusion::at_c<1>(ze)))(i,j,k) );
            }
        };

        struct increment {
            template <typename Iterator>
            GT_FUNCTION
            void operator()(Iterator & it) const {
                ++it;
            }
        };

    } // namespace iterate_domain_aux 

    template <typename LocalDomain>
    struct iterate_domain {
        typedef typename LocalDomain::local_iterators_type local_iterators_type;

        size_t stride;
        LocalDomain const& local_domain;
        mutable local_iterators_type local_iterators;

        iterate_domain(LocalDomain const& local_domain, int i, int j, int k)
            : stride(1)
            , local_domain(local_domain)
        {
            typedef boost::fusion::vector<local_iterators_type&, typename LocalDomain::local_args_type const&> to_zip;
            typedef boost::fusion::zip_view<to_zip> zipping;
            to_zip z(local_iterators, local_domain.local_args);
            boost::fusion::for_each(zipping(z), iterate_domain_aux::assign_iterators(i,j,k));
        }

        void increment() const {
            boost::fusion::for_each(local_iterators, iterate_domain_aux::increment());
        }

        template <typename T>
        GT_FUNCTION
        void info(T const &x) const {
            local_domain.info(x);
        }

        template <typename T>
        GT_FUNCTION
        typename boost::mpl::at<typename LocalDomain::esf_args, typename T::index_type>::type::value_type&  
        operator()(T const& ) const {
            return *(boost::fusion::at<typename T::index_type>(local_iterators));
        }

        template <typename T>
        GT_FUNCTION
        typename boost::mpl::at<typename LocalDomain::esf_args, typename T::index>::type::value_type const& 
        operator()(T const& , int i, int j, int k) const {
            int offset = (boost::fusion::at<typename T::index_type>(local_domain.local_args))->offset(i,j,k);
            return &(boost::fusion::at<typename T::index_type>(local_iterators)+offset);
        }

    };

} // namespace gridtools
