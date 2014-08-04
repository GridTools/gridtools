#pragma once

#include "../storage/host_tmp_storage.h"

namespace gridtools {
    namespace _impl {
        struct instantiate_tmps {
            int tileK;

            GT_FUNCTION
            instantiate_tmps(int tileK)
                : tileK(tileK)
            {}

            // ElemType: an element in the data field place-holders list
            template <typename ElemType>
            GT_FUNCTION
            void operator()(ElemType*&  e) const {

#ifndef __CUDACC__
                std::string s = boost::lexical_cast<std::string>(ElemType::minusi::value)+
                    boost::lexical_cast<std::string>(ElemType::minusj::value)+
                    boost::lexical_cast<std::string>(ElemType::plusi::value)+
                    boost::lexical_cast<std::string>(ElemType::plusj::value);
#endif
                typename ElemType::value_type x = 5.7;
                e = new ElemType(
                                 tileK,
                                 typename ElemType::value_type(),
#ifndef __CUDACC__
                                 s);
#else
                );
#endif
            }
        };

        struct delete_tmps {
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
#ifndef __CUDA_ARCH__
                delete elem;
#endif
            }
        };

    } // namespace _impl

template <typename Backend>
struct heap_allocated_temps {
        /**
         * This function is to be called by intermediate representation or back-end
         *
         * @tparam MssType The multistage stencil type as passed to the back-end
         * @tparam RangeSizes mpl::vector with the sizes of the extents of the
         *         access for each functor listed as linear_esf in MssType
         * @tparam Domain The user domain type - Deduced from argument list
         * @tparam coords The user coordinates type - Deduced from argument list
         */
        template <typename ArgList, typename Coords>
        static void prepare_temporaries(ArgList & arg_list, Coords const& coords) {
            int tileK = coords.value_at_top()-coords.value_at_bottom();

#ifndef NDEBUG
            std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

            boost::fusion::for_each(arg_list, _impl::instantiate_tmps(tileK));

            //            domain.is_ready = true;
        }

        /**
           This function calls d2h_update on all storages, in order to
           get the data back to the host after a computation.
        */
        template <typename Domain>
        static void finalize_computation(Domain & domain) {
            //domain.finalize_computation();
        }

    };
} // namespace gridtools
