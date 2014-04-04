#pragma once

#include "host_device.h"
#include "gpu_clone.h"
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include "iterate_domain.h"

namespace gridtools {

    struct outout {
        template <typename t>
        void operator()(t const&) const {
            std::cout << "outout " << t() << std::endl;
        }
    };

    struct printhem {
        template <typename t>
        void operator()(t const& x) const {
            std::cout << "outout whowho" << x << std::endl;
        }
    };

    struct outouts {
        template <typename t>
        void operator()(t const&) const {
            boost::remove_pointer<t>::type::text();
        }
    };

    namespace local_domain_aux {
        template <typename List, typename Index>
        struct get_index {
            typedef typename boost::mpl::at<
                List, 
                Index
                >::type type;
        };

        struct get_storage {
            template <typename U>
            struct apply {
                typedef typename U::storage_type* type;
            };
        };

        struct get_iterator {
            template <typename U>
            struct apply {
                typedef typename U::iterator_type type;
            };
        };

        template <typename Domain>
        struct assign_base_pointers {

            Domain const& domain;

            assign_base_pointers(Domain const& domain)
                : domain(domain)
            {}

            template <typename ZipElem>
            void operator()(ZipElem const& ze) const {
                typedef typename boost::remove_reference<typename boost::fusion::result_of::at_c<ZipElem, 0>::type>::type::index_type index;
                // typedef typename boost::remove_pointer<typename boost::remove_reference<typename boost::fusion::result_of::at<typename Domain::arg_list, index>::type>::type>::type storage_type;

                // std::cout << "outout pointer " 
                //           << index()
                //           << " " << boost::fusion::size(domain.storage_pointers) << " "
                //           << " " << boost::fusion::at<index>(domain.storage_pointers) << std::endl;;
                //                storage_type::text();
                boost::fusion::at_c<1>(ze) = 
                    boost::fusion::at<index>(domain.storage_pointers);
            }
        };

    } // namespace gt_aux



    /**
     * This is the base class for local_domains to extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. There is one version which provide coordinates to the functor
     * and one that does not
     * 
     * @tparam EsfDescriptor The descriptor of the elementary stencil function
     * @tparam Domain The full domain type
     */
    template <typename Derived, typename EsfDescriptor, typename Domain>
    struct local_domain_base: public clonable_to_gpu<Derived> {

        typedef local_domain_base<Derived, EsfDescriptor, Domain> this_type;

        typedef typename EsfDescriptor::args esf_args;
        typedef typename EsfDescriptor::esf_function esf_function;
        typedef typename Domain::placeholders dom_placeholders;


        typedef boost::mpl::range_c<int, 0, boost::mpl::size<esf_args>::type::value > the_range;
        typedef typename boost::mpl::fold<the_range,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              local_domain_aux::get_index<esf_args,  boost::mpl::_2>
                                              >
                                          >::type domain_indices;
    
        typedef typename boost::mpl::transform<domain_indices,
                                               local_domain_aux::get_storage
                                               >::type mpl_storages;

        typedef typename boost::mpl::transform<domain_indices,
                                               local_domain_aux::get_iterator
                                               >::type mpl_iterators;

        typedef typename boost::fusion::result_of::as_vector<mpl_storages>::type local_args_type;
        typedef typename boost::fusion::result_of::as_vector<mpl_iterators>::type local_iterators_type;
                                          
        typedef iterate_domain<this_type> iterate_domain_type;

        local_args_type local_args;

        typedef Domain domain_type;

        Domain *dom;
        Domain *g_dom;

        //int m_i,m_j,m_k;

        template <typename Dom, typename IsActuallyClonable, int DUMMY = 0>
        struct pointer_if_clonable {
            static Dom* get(Dom* d) {
                return d;
            }
        };

        template <typename Dom, int DUMMY>
        struct pointer_if_clonable<Dom, boost::true_type, DUMMY> {
            static Dom* get(Dom* d) {
                return d->gpu_object_ptr;
            }
        };


        local_domain_base() {}
                    
        GT_FUNCTION
        void init(Domain* _dom) 
        {
            gridtools::for_each<domain_indices>(outout());
            gridtools::for_each<mpl_storages>(outouts());

            typedef boost::fusion::vector<domain_indices const&, local_args_type&> to_zip;
            typedef boost::fusion::zip_view<to_zip> zipping;

            to_zip z(domain_indices(), local_args);

            dom = _dom;
            g_dom = pointer_if_clonable<Domain, typename Domain::actually_clonable>::get(_dom);

            boost::fusion::for_each(zipping(z), local_domain_aux::assign_base_pointers<Domain>(*g_dom));

        }

        __device__
        local_domain_base(local_domain_base const& other)
            : dom(other.g_dom)
            // , m_i(other.m_i)
            // , m_j(other.m_j)
            // , m_k(other.m_k)
        { }

        template <typename T>
        GT_FUNCTION
        typename boost::mpl::at<esf_args, typename T::index_type>::type::value_type&  
        operator()(T const& t) const {
            return dom->template direct<typename boost::mpl::at<esf_args, typename T::index_type>::type::index_type>(/*typename T::index()*/);
        }

        template <typename T>
        GT_FUNCTION
        typename boost::mpl::at<esf_args, typename T::index>::type::value_type const& 
        operator()(T const&, int i, int j, int k) const {
            return dom->template direct<typename boost::mpl::at<esf_args, typename T::index>::type::index>();
        }

        template <typename T>
        GT_FUNCTION
        typename boost::fusion::result_of::at<esf_args, typename T::index>::value_type& 
        get(int i, int j, int k) const {
            return dom->template direct<typename boost::mpl::at_c<esf_args, T::index>::type::index>();     
        }

        template <typename T>
        GT_FUNCTION
        typename boost::fusion::result_of::at<esf_args, typename T::index>::value_type& 
        operator[](T const&) const {
            return dom->template direct<boost::mpl::at_c<esf_args, T::index>::type::index>();
        }

        GT_FUNCTION
        void move_to(int i, int j, int k) const {
            //printf("ADDR-- %X\n", dom);
            dom->move_to(i,j,k);
            //info();
        }

        GT_FUNCTION
        void increment() const {
            dom->template increment_along<2>();
        }

        template <typename T>
        void info(T const&) const {
            T::info();
            std::cout << "[" << boost::mpl::at_c<esf_args, T::index_type::value>::type::index_type::value << "] ";
            dom->template storage_info<typename boost::mpl::at_c<esf_args, T::index_type::value>::type::index_type>();
            //            typename boost::mpl::at<esf_args, typename T::index_type>::type::value_type&  
        }

        GT_FUNCTION
        void info() const {
            dom->info();
        }
    };

    //            template <typename EsfDescriptor, typename Domain>
    //            struct local_domain_location : public local_domain_base<EsfDescriptor, Domain> {
    //                typedef local_domain_base<EsfDescriptor, Domain> base_type;
    //                typedef typename EsfDescriptor::args esf_args;
    //                typedef typename EsfDescriptor::esf_function esf_function;
    //                typedef typename Domain::placeholders dom_placeholders;
    //                //typedef typename Domain::arg dom_args;
    //                typedef Domain domain_type;
    //
    //                int m_i,m_j,m_k;
    //
    //                explicit local_domain_location(Domain const & dom, int i, int j, int k)
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
     * @tparam EsfDescriptor The descriptor of the elementary stencil function
     * @tparam Domain The full domain type
     */
    template <typename EsfDescriptor, typename Domain>
    struct local_domain : public local_domain_base< local_domain<EsfDescriptor, Domain>, EsfDescriptor, Domain> {
        typedef local_domain_base<local_domain<EsfDescriptor, Domain>, EsfDescriptor, Domain> base_type;
        typedef typename EsfDescriptor::args esf_args;
        typedef typename EsfDescriptor::esf_function esf_function;
        typedef typename Domain::placeholders dom_placeholders;
        typedef Domain domain_type;

        GT_FUNCTION
        local_domain() {}
                
        __device__
        local_domain(local_domain const& other)
            : base_type(other)
        {}

        GT_FUNCTION
        void init(Domain* dom, int, int, int)
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
