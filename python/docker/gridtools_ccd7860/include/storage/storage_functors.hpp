#pragma once
#include "base_storage.hpp"
#include "host_tmp_storage.hpp"

namespace gridtools {

template<typename DestCont, typename OrigCont>
struct copy_pointers_functor {

    copy_pointers_functor(DestCont &dc, OrigCont& oc) : m_dc(dc), m_oc(oc) {}

    template <typename Index>
    GT_FUNCTION_WARNING
    void operator()(const Index& ) const {
        assign<Index>();
    }
private:
    //do not copy pointers in case storage is a temporary
    template<typename Index>
    GT_FUNCTION_WARNING
    void assign(typename boost::enable_if_c<
            is_temporary_storage<
                typename boost::mpl::at<DestCont, Index>::type
            >::value
        >::type* = 0) const {}

    template<typename Index>
    GT_FUNCTION_WARNING
    void assign(typename boost::disable_if_c<
            is_temporary_storage<
                typename boost::mpl::at<DestCont, Index>::type
            >::value
        >::type* = 0) const
    {
        boost::fusion::at<Index>(m_dc) = boost::fusion::at<Index>(m_oc);
    }

    DestCont& m_dc;
    OrigCont& m_oc;
};

/**@brief Functor updating the pointers on the device */
        struct update_pointer {
#ifdef __CUDACC__

            template < typename StorageType//typename T, typename U, bool B
                       >
            GT_FUNCTION_WARNING
            void operator()(/*base_storage<enumtype::Cuda,T,U,B
                              >*/StorageType *& s) const {
                if (s) {
                    copy_data_impl<StorageType>(s);

                    s->clone_to_gpu();
                }
            }

        private:
            //we do not copy data into the gpu in case of a temporary
            template<typename StorageType>
            GT_FUNCTION_WARNING
            void copy_data_impl(StorageType *& s,
                    typename boost::enable_if_c<is_host_tmp_storage<StorageType>::value>::type* = 0) const
            {}

            template<typename StorageType>
            GT_FUNCTION_WARNING
            void copy_data_impl(StorageType *& s,
                    typename boost::disable_if_c<is_host_tmp_storage<StorageType>::value>::type* = 0) const
            {
                s->copy_data_to_gpu();
            }

#else
            template <typename StorageType>
            GT_FUNCTION_WARNING
            void operator()(StorageType* s) const {}
#endif
        };


        //TODO : This struct is never used
        struct call_h2d {
            template <typename StorageType>
            GT_FUNCTION
            void operator()(StorageType * arg) const {
#ifndef __CUDA_ARCH__
                do_impl<StorageType>(arg,
                        static_cast<typename is_no_storage_type_yet<StorageType>::type*>(0)
                );
#endif
            }
        private:
            template <typename StorageType>
            GT_FUNCTION
            void do_impl(StorageType * arg,
                    typename boost::enable_if_c<is_no_storage_type_yet<StorageType>::value>::type* = 0) const {}
            template <typename StorageType>
            GT_FUNCTION
            void do_impl(StorageType * arg,
                    typename boost::disable_if_c<is_no_storage_type_yet<StorageType>::value>::type* = 0) const {
                arg->h2d_update();
            }
        };

        struct call_d2h {
            template <typename StorageType>
            GT_FUNCTION
            void operator()(StorageType * arg) const {
#ifndef __CUDA_ARCH__
                do_impl<StorageType>(arg,
                        static_cast<typename is_no_storage_type_yet<StorageType>::type*>(0)
                );
#endif
            }
        private:
            template <typename StorageType>
            GT_FUNCTION
            void do_impl(StorageType * arg,
                    typename boost::enable_if_c<is_no_storage_type_yet<StorageType>::value>::type* = 0) const {}
            template <typename StorageType>
            GT_FUNCTION
            void do_impl(StorageType * arg,
                    typename boost::disable_if_c<is_no_storage_type_yet<StorageType>::value>::type* = 0) const {
                arg->d2h_update();
            }
        };

} // namespace gridtools
