#pragma once

#include "../storage/host_tmp_storage.hpp"
#include "backend_fwd.hpp"
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/filter_view.hpp>

/**
@file
TODO Document me!
*/

namespace gridtools {
    namespace _impl {

        /** prepare temporaries struct, constructing the domain for the temporary fields, with the arguments
            to the constructor depending on the specific strategy */
        template <typename ArgList, typename Coords, typename BackendType>
        struct prepare_temporaries_functor;

        /**
           Specialization for Naive policy
         */
        template <typename ArgList, typename Coords,
                  enumtype::backend BackendId>
        struct prepare_temporaries_functor<ArgList, Coords, backend<BackendId, enumtype::/*strategy::*/Naive> >
        {
            /**
               @brief instantiate the \ref gridtools::domain_type for the temporary storages
            */
            struct instantiate_tmps
            {
                uint_t m_tile_i;// tile along i
                uint_t m_tile_j;// tile along j
                uint_t m_tile_k;// tile along k

                GT_FUNCTION
                instantiate_tmps(uint_t tile_i, uint_t tile_j, uint_t tile_k)
                    : m_tile_i(tile_i)
                    , m_tile_j(tile_j)
                    , m_tile_k(tile_k)
                {}

                // ElemType: an element in the data field place-holders list
                template <typename ElemType>
                void operator()(ElemType*&  e) const {
                    //ElemType::info_string.c_str();

                    //calls the constructor of the storage
                    //TODO noone deletes this new
                    e = new ElemType(m_tile_i,
                                     m_tile_j,
                                     m_tile_k);
                    e->set_name("default tmp storage");
                }
            };

            static void prepare_temporaries(ArgList & arg_list, Coords const& coords) {

#ifdef __VERBOSE__
                std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

                typedef boost::fusion::filter_view<ArgList,
                    is_temporary_storage<boost::mpl::_1> > view_type;

                view_type fview(arg_list);

                boost::fusion::for_each(fview,
                                        instantiate_tmps(coords.direction_i().total_length(),
                                                         coords.direction_j().total_length(),
                                                         coords.value_at_top()-coords.value_at_bottom()+1));

            }

        };

        /**
           Specialization for Block policy
         */
        template <typename ArgList, typename Coords,
                  enumtype::backend BackendId>
        struct prepare_temporaries_functor<ArgList, Coords, backend<BackendId, enumtype::/*strategy::*/Block> >
        {

            typedef backend<BackendId, enumtype/*::strategy*/::Block> backend_type;
            /**
               @brief instantiate the \ref gridtools::domain_type for the temporary storages
            */
            struct instantiate_tmps
            {
                uint_t m_offset_i;// offset along i
                uint_t m_offset_j;// offset along j
                uint_t m_offset_k;// offset along k
                uint_t m_n_i_threads;
                uint_t m_n_j_threads;

                GT_FUNCTION
                instantiate_tmps(uint_t offset_i,
                                 uint_t offset_j,
                                 uint_t offset_k,
                                 uint_t m_n_i_threads,
                                 uint_t m_n_j_threads)
                    : m_offset_i(offset_i)
                    , m_offset_j(offset_j)
                    , m_offset_k(offset_k)
                    , m_n_i_threads(m_n_i_threads)
                    , m_n_j_threads(m_n_j_threads)
                {}

                // ElemType: an element in the data field place-holders list
                template <typename ElemType>
                void operator()(ElemType*&  e) const {
                    //char const* s = "default tmp storage";//avoids a warning
                    //ElemType::info_string.c_str();

                    //calls the constructor of the storage
                    //TODO noone deletes this new
                    e = new ElemType(m_offset_i,
                                     m_offset_j,
                                     m_offset_k,
                                     m_n_i_threads,
                                     m_n_j_threads);
                }
            };

            static void prepare_temporaries(ArgList & arg_list, Coords const& coords) {
                //static const enumtype::strategy StrategyType = Block;

#ifdef __VERBOSE__
                std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

                typedef boost::fusion::filter_view<ArgList,
                    is_temporary_storage<boost::mpl::_1> > view_type;

                view_type fview(arg_list);
                boost::fusion::for_each(fview,
                                        instantiate_tmps
                                        ( coords.i_low_bound(),
                                          coords.j_low_bound(),
                                          coords.value_at_top()-coords.value_at_bottom()+1,
                                          backend_type::n_i_pes()(coords.i_high_bound() - coords.i_low_bound()),
                                          backend_type::n_j_pes()(coords.j_high_bound() - coords.j_low_bound())
                                         )
                                        );
            }

        };

        // noone calls this!!!
        // I know! we should try to put this back, I had issues with double frees at some point
        struct delete_tmps {
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
                delete elem;
            }
        };

    } // namespace _impl

} // namespace gridtools
