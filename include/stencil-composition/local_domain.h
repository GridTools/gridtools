#pragma once

#include "../common/host_device.h"
#include "../common/gpu_clone.h"
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/utility.hpp>

#include "iterate_domain.h"

namespace gridtools {

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

        template <typename ArgList>
        struct assign_storage_pointers {

            ArgList const& arg_list;

            GT_FUNCTION_WARNING
            assign_storage_pointers(ArgList const& arg_list)
                : arg_list(arg_list)
            {}

            template <typename ZipElem>
            GT_FUNCTION_WARNING
            void operator()(ZipElem const& ze) const {
                typedef typename boost::remove_reference<typename boost::fusion::result_of::at_c<ZipElem, 0>::type>::type::index_type index;

                boost::fusion::at_c<1>(ze) =
                    boost::fusion::at<index>(arg_list);
            }
        };


        /** Just extract the storage types. In case of temporaries, these types
            are the storage types containing the storage classes that contains
            the "repositories" of all the perthread containers.
        */
        template <typename StorageList>
        struct extract_types {
            template <typename ElemType>
            struct apply {
                typedef typename boost::remove_reference<
                    typename boost::fusion::result_of::at<StorageList, typename ElemType::index_type>::type
                    >::type type;
            };
        };

        /** Just extract the storage types. In case of temporaries, these types
            are the storage types containing the actual storage types used by the
            individual threads. This requires a difference w.r.t. extract_types
            for how to deal with temporaries.

            Since certain modifications happend this metafunction is actually
            identical, in behavior, with extract_types. 
        */
        template <typename StorageList>
        struct extract_actual_types {

            template <typename Storage, typename Enable=void>
            struct check_if_temporary;

            template <typename Storage>
            struct check_if_temporary<Storage, typename boost::enable_if_c<is_temporary_storage<Storage>::value>::type> {
                typedef Storage type;
            };

            template <typename Storage>
            struct check_if_temporary<Storage, typename boost::disable_if_c<is_temporary_storage<Storage>::value>::type> {
                typedef Storage type;
            };


            template <typename ElemType>
            struct apply {
                typedef typename check_if_temporary<
                    typename boost::remove_reference<
                        typename boost::fusion::result_of::at<StorageList, typename ElemType::index_type>::type
                        >::type
                    >::type type;
            };
        };

    } // namespace gt_aux


    template <bool IsStateful, typename T>
    struct select_iterate_domain;

    template <typename T>
    struct select_iterate_domain<true, T> {
        typedef stateful_iterate_domain<T> type;
    };

    template <typename T>
    struct select_iterate_domain<false, T> {
        typedef iterate_domain<T> type;
    };

    /**
     * This is the base class for local_domains to extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. There is one version which provide coordinates to the functor
     * and one that does not
     *
     * @tparam EsfDescriptor The descriptor of the elementary stencil function
     * @tparam Domain The full domain type
     */
    template <typename Derived, typename StoragePointers, bool IsStateful, typename EsfDescriptor>
    struct local_domain_base: public clonable_to_gpu<Derived> {

        typedef local_domain_base<Derived, StoragePointers, IsStateful, EsfDescriptor> this_type;

        typedef typename EsfDescriptor::args esf_args;
        typedef typename EsfDescriptor::esf_function esf_function;


        typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<esf_args>::type::value > the_range;

        typedef typename boost::mpl::fold<the_range,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              local_domain_aux::get_index<esf_args,  boost::mpl::_2>
                                              >
                                          >::type domain_indices;

        /** creates a vector of storage types from the StoragePointers sequence */
        typedef typename boost::mpl::fold<domain_indices,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              typename local_domain_aux::extract_types<
                                                  StoragePointers>::template apply<boost::mpl::_2>
                                              >
                                          >::type mpl_storages;

        /** creates a vector of storage types from the StoragePointers sequence */
        typedef typename boost::mpl::fold<domain_indices,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              typename local_domain_aux::extract_actual_types<
                                                  StoragePointers>::template apply<boost::mpl::_2>
                                              >
                                          >::type mpl_actual_storages;

        typedef typename boost::fusion::result_of::as_vector<mpl_storages>::type local_args_type;
        typedef typename boost::fusion::result_of::as_vector<mpl_actual_storages>::type actual_args_type;


        typedef typename select_iterate_domain<IsStateful, this_type>::type iterate_domain_t;

        local_args_type local_args;

        template <typename Dom, typename IsActuallyClonable, uint_t DUMMY = 0>
        struct pointer_if_clonable {
            static Dom* get(Dom* d) {
                return d;
            }
        };

        template <typename Dom, uint_t DUMMY>
        struct pointer_if_clonable<Dom, boost::true_type, DUMMY> {
            static Dom* get(Dom* d) {
                return d->gpu_object_ptr;
            }
        };

        GT_FUNCTION_WARNING
        local_domain_base() {}

        template <typename Domain, typename ActualArgs>
        GT_FUNCTION
        void init(Domain const& _dom, ActualArgs const& actual_args)
        {
            typedef boost::fusion::vector<domain_indices const&, local_args_type&> to_zip;
            typedef boost::fusion::zip_view<to_zip> zipping;

            to_zip z(domain_indices(), local_args);

            boost::fusion::for_each(zipping(z), local_domain_aux::assign_storage_pointers<ActualArgs>(actual_args));

        }

        __device__
        local_domain_base(local_domain_base const& other)
            : local_args(other.local_args)
        { }

        template <typename T>
        void info(T const&) const {
            T::info();
            std::cout << "[" << boost::mpl::at_c<esf_args, T::index_type::value>::type::index_type::value << "] ";
        }

        struct show_local_args_info {
            template <typename T>
            void operator()(T const & e) const {
                e->info();
            }
        };

        GT_FUNCTION
        void info() const {
            std::cout << "        -----v SHOWING LOCAL ARGS BELOW HERE v-----" << std::endl;
            boost::fusion::for_each(local_args, show_local_args_info());
            std::cout << "        -----^ SHOWING LOCAL ARGS ABOVE HERE ^-----" << std::endl;
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
    template <typename StoragePointers, typename EsfDescriptor, bool IsStateful>
    struct local_domain : public local_domain_base< local_domain<StoragePointers,EsfDescriptor,IsStateful>, StoragePointers, IsStateful, EsfDescriptor > {
        typedef local_domain_base<local_domain<StoragePointers,EsfDescriptor,IsStateful>, StoragePointers, IsStateful, EsfDescriptor > base_type;
        typedef EsfDescriptor esf_descriptor;
        typedef StoragePointers storage_pointers;
        //typedef Iterators iterators;
        typedef typename EsfDescriptor::args esf_args;
        typedef typename EsfDescriptor::esf_function esf_function;

        GT_FUNCTION
        local_domain() {}

        __device__
        local_domain(local_domain const& other)
            : base_type(other)
        {}

        template <typename Domain, typename ArgList>
        GT_FUNCTION
        void init(Domain const& dom, ArgList const& arg_list, uint_t, uint_t, uint_t)
        {
            base_type::init(dom, arg_list);
#ifndef NDEBUG
#ifndef __CUDACC__
            std::cout << "LOCAL DOMAIN" << std::endl;
#endif
#endif
        }

/**stub methods*/
        GT_FUNCTION
        uint_t i() const {return 1e9; }
        GT_FUNCTION
        uint_t j() const {return 1e9; }
        GT_FUNCTION
        uint_t k() const {return 1e9; }
    };

    template <typename StoragePointers, typename EsfDescriptor, bool IsStateful>
    std::ostream& operator<<(std::ostream& s, local_domain<StoragePointers, EsfDescriptor, IsStateful> const&) {
        return s << "local_domain<stuff>";
    }

    template<typename T> struct is_local_domain : boost::mpl::false_{};

    template <typename StoragePointers, typename EsfDescriptor, bool IsStateful>
    struct is_local_domain<local_domain<StoragePointers, EsfDescriptor, IsStateful> > : boost::mpl::true_{};

}
