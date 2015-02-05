#pragma once

#include "../storage/host_tmp_storage.h"
#include "backend_traits.h"
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/filter_view.hpp>

namespace gridtools {
    namespace _impl {


/**
@brief instantiate the \ref gridtools::domain_type for the temporary storages
*/
        struct instantiate_tmps
        {
            uint_t m_tile_i;// or offset along i
            uint_t m_tile_j;// or offset along j
            uint_t m_tile_k;// or offset along k

            GT_FUNCTION
            instantiate_tmps(uint_t tile_i, uint_t tile_j, uint_t tile_k)
                :
                m_tile_i(tile_i)
                , m_tile_j(tile_j)
                , m_tile_k(tile_k)
            {}

            // ElemType: an element in the data field place-holders list
            template <typename ElemType>
            void operator()(ElemType*&  e) const {
                char const* s = "default tmp storage";//avoids a warning
                //ElemType::info_string.c_str();

                //calls the constructor of the storage
                //TODO noone deletes this new
                e = new ElemType(m_tile_i,
                                 m_tile_j,
                                 m_tile_k,
//                                 0, // offset in k is zero for now
                                 typename ElemType::value_type(),
                                 s);
            }
    };


        // noone calls this!!!
        struct delete_tmps {
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
#ifndef __CUDA_ARCH__
                delete elem;
#endif
            }
        };

        namespace{
            using namespace enumtype;
            template< strategy Str >
                struct policy;
        }//unnamed namespace

/** prepare temporaries struct, constructing the domain for the temporary fields, with the arguments to the constructor depending on the specific strategy */
    template <typename ArgList, typename Coords, enumtype::strategy StrategyType>
    struct prepare_temporaries_functor
    {
        static void prepare_temporaries(ArgList & arg_list, Coords const& coords) {

#ifndef NDEBUG
            std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

            typedef boost::fusion::filter_view<ArgList,
                                               is_temporary_storage<boost::mpl::_1> > view_type;

            view_type fview(arg_list);

            boost::fusion::for_each(fview, _impl::instantiate_tmps( policy<StrategyType>::value_i(coords), policy<StrategyType>::value_j(coords), policy<StrategyType>::value_k(coords)));

        }

    };

        namespace{
/**Policy for the \ref gridtools::domain_type constructor arguments. When the Block strategy is chosen the arguments value_i and value_j represent an offset index in the i and j dimensions. */
        template<>
        struct policy<Block>
            {
                template <typename Coords>
                static uint_t value_k(Coords& coords){ return coords.value_at_top()-coords.value_at_bottom()+1;}
                template <typename Coords>
                static uint_t value_i(Coords& coords){ return coords.i_low_bound();}
                template <typename Coords>
                static uint_t value_j(Coords& coords){ return coords.j_low_bound();}
        };

/**Policy for the \ref gridtools::domain_type constructor arguments. When the Naive strategy is chosen the arguments value_i and value_j represent the total number of indices in the i and j directions. */
        template<>
        struct policy<Naive>
            {
                template <typename Coords>
                static uint_t value_k(Coords& coords){ return coords.value_at_top()-coords.value_at_bottom()+1;}
                template <typename Coords>
                static uint_t value_i(Coords& coords){ return coords.direction_i().total_length();}
                template <typename Coords>
                static uint_t value_j(Coords& coords){ return coords.direction_j().total_length();}
        };

        }

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
    struct printpointers {
        template <typename T>
        void operator()(T const& p) const {
            std::cout << std::hex << p << std::dec << std::endl;
        }
    };


    };

} // namespace gridtools
