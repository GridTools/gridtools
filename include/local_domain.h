#pragma once

#include "host_device.h"

namespace gridtools {

    /**
     * This is the base class for local_domains to extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. There is one version which provide coordinates to the functor
     * and one that does not
     * 
     * @tparam t_esf_descriptor The descriptor of the elementary stencil function
     * @tparam t_domain The full domain type
     */
    template <typename t_esf_descriptor, typename t_domain>
    struct local_domain_base {
        typedef typename t_esf_descriptor::args esf_args;
        typedef typename t_esf_descriptor::esf_function esf_function;
        typedef typename t_domain::placeholders dom_placeholders;
        typedef t_domain domain_type;

        t_domain *dom;

        int m_i,m_j,m_k;

                    
        GT_FUNCTION
        void init(t_domain* _dom) {
            dom = _dom;
        }

        template <typename T>
        GT_FUNCTION
        typename boost::mpl::at<esf_args, typename T::index_type>::type::value_type&  
        operator()(T const& t) const {
            return dom->template direct<typename boost::mpl::template at<esf_args, typename T::index_type>::type::index_type>(/*typename T::index()*/);
        }

        template <typename T>
        GT_FUNCTION
        typename boost::mpl::at<esf_args, typename T::index>::type::value_type& 
        operator()(T const&, int i, int j, int k) const {
            return dom->template direct<typename boost::mpl::template at<esf_args, typename T::index>::type::index>();
        }

        template <typename T>
        GT_FUNCTION
        typename boost::fusion::result_of::at<esf_args, typename T::index>::value_type& 
        get(int i, int j, int k) const {
            return dom->template direct<typename boost::mpl::template at_c<esf_args, T::index>::type::index>();     
        }

        template <typename T>
        GT_FUNCTION
        typename boost::fusion::result_of::at<esf_args, typename T::index>::value_type& 
        operator[](T const&) const {
            return dom->template direct<boost::mpl::template at_c<esf_args, T::index>::type::index>();
        }

        GT_FUNCTION
        void move_to(int i, int j, int k) const {
            dom->move_to(i,j,k);
        }

        GT_FUNCTION
        void increment() const {
            dom->template increment_along<2>();
        }

    };

    //            template <typename t_esf_descriptor, typename t_domain>
    //            struct local_domain_location : public local_domain_base<t_esf_descriptor, t_domain> {
    //                typedef local_domain_base<t_esf_descriptor, t_domain> base_type;
    //                typedef typename t_esf_descriptor::args esf_args;
    //                typedef typename t_esf_descriptor::esf_function esf_function;
    //                typedef typename t_domain::placeholders dom_placeholders;
    //                //typedef typename t_domain::arg dom_args;
    //                typedef t_domain domain_type;
    //
    //                int m_i,m_j,m_k;
    //
    //                explicit local_domain_location(t_domain const & dom, int i, int j, int k)
    //                    : base_type(dom)
    //                    , m_i(i)
    //                    , m_j(j)
    //                    , m_k(k)
    //                {
    //                    std::cout << "LOCAL DOMAIN LOCATIONNNNN" << std::endl;
    //                }
    //
    //                int i() const { return m_i;}
    //                int j() const { return m_j;}
    //                int k() const { return m_k;}
    //            };

    /**
     * This class extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. This version does not provide coordinates
     * to the function operator
     * 
     * @tparam t_esf_descriptor The descriptor of the elementary stencil function
     * @tparam t_domain The full domain type
     */
    template <typename t_esf_descriptor, typename t_domain>
    struct local_domain : public local_domain_base<t_esf_descriptor, t_domain> {
        typedef local_domain_base<t_esf_descriptor, t_domain> base_type;
        typedef typename t_esf_descriptor::args esf_args;
        typedef typename t_esf_descriptor::esf_function esf_function;
        typedef typename t_domain::placeholders dom_placeholders;
        typedef t_domain domain_type;

        GT_FUNCTION
        local_domain() {}
                
        GT_FUNCTION
        void init(t_domain* dom, int, int, int)
        {
            base_type::init(dom);
#ifndef NDEBUG
#ifndef __CUDACC__
            std::cout << "LOCAL DOMAIN" << std::endl;
#endif
#endif
        }

        GT_FUNCTION
        int i() const {return -1; }
        GT_FUNCTION
        int j() const {return -1; }
        GT_FUNCTION
        int k() const {return -1; }
    };
}
