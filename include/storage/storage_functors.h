#pragma once
#include "base_storage.h"
#include "host_tmp_storage.h"

namespace gridtools {

template<typename DestCont, typename OrigCont>
struct copy_pointers_functor {

    copy_pointers_functor(DestCont &dc, OrigCont& oc) : m_dc(dc), m_oc(oc) {}

    template <typename Index>
    GT_FUNCTION_WARNING
    void operator()(const Index& ) const {
        printf("CopyPointer\n");

        assign<Index>(static_cast<typename is_temporary_storage< typename boost::mpl::at<DestCont, Index>::type >::type*> (0) );
    }
private:
    template<typename Index>
    GT_FUNCTION_WARNING
    void assign(boost::mpl::bool_<true>*) const
    {
        printf("NoCopyPointer\n");

    }
    template<typename Index>
    GT_FUNCTION_WARNING
    void assign(boost::mpl::bool_<false>*) const
    {
        printf("YesCopyPointer\n");

        boost::fusion::at<Index>(m_dc) = boost::fusion::at<Index>(m_oc);
    }

    DestCont& m_dc;
    OrigCont& m_oc;
};

/**@brief Functor updating the pointers on the device */
        struct update_pointer {
#ifdef __CUDACC__
            template<typename T> struct printy{BOOST_MPL_ASSERT_MSG((false), YYYYYYYYYYYYYYYY, (T));};

            template < typename StorageType//typename T, typename U, bool B
                       >
            GT_FUNCTION_WARNING
            void operator()(/*base_storage<enumtype::Cuda,T,U,B
                              >*/StorageType *& s) const {
                printf("UpdatePointer\n");
                do_impl<StorageType>(s,
                        static_cast<typename is_host_tmp_storage<StorageType>::type*>(0)
                );
            }

        private:
            template<typename StorageType>
            GT_FUNCTION_WARNING
            void do_impl(StorageType *& s, boost::mpl::bool_<true>*) const
            {
            }
            template<typename StorageType>
            GT_FUNCTION_WARNING
            void do_impl(StorageType *& s, boost::mpl::bool_<false>*) const
            {
                printf("UpdatePointerNonTemp\n");

                if (s) {
                    s->copy_data_to_gpu();
                    s->clone_to_gpu();
                    s = s->gpu_object_ptr;
                    printf("Copy F\n");
                }
            }

#else
            template <typename StorageType>
            GT_FUNCTION_WARNING
            void operator()(StorageType* s) const {}
#endif
        };

        //TODO : This struct is never used
        struct call_h2d {
            template <typename Arg>
            GT_FUNCTION
            void operator()(Arg * arg) const {
#ifndef __CUDA_ARCH__
                arg->h2d_update();
#endif
            }
        };

        struct call_d2h {
            template <typename StorageType>
            GT_FUNCTION
            void operator()(StorageType * arg) const {
                printf("MY \n ");

#ifndef __CUDA_ARCH__
                do_impl<StorageType>(arg,
                        static_cast<typename is_no_storage_type_yet<StorageType>::type*>(0)
                );
#endif
            }
        private:
            template <typename StorageType>
            GT_FUNCTION
            void do_impl(StorageType * arg, boost::mpl::bool_<true>*) const {
                printf("SIs \n ");

            }
            template <typename StorageType>
            GT_FUNCTION
            void do_impl(StorageType * arg, boost::mpl::bool_<false>*) const {
                printf("NO \n ");
                arg->d2h_update();
            }

        };

} // namespace gridtools
