#pragma once

#include "common/defs.hpp"
#include "stencil-composition/backend_fwd.hpp"
#include "common/is_temporary_storage.hpp"
#include "storage/base_storage.hpp"
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/filter_view.hpp>
#include "stencil-composition/grid.hpp"
#include "storage/metadata_set.hpp"

/**
@file
TODO Document me!
*/

namespace gridtools {
    namespace _impl {

        /** prepare temporaries struct, constructing the domain for the temporary fields, with the arguments
            to the constructor depending on the specific strategy */
        template <typename ArgList,
                  typename MetaList,
                  typename Grid,
                  enumtype::platform BackendId,
                  enumtype::strategy StrategyId>
        struct prepare_temporaries_functor;

        /**
           Specialization for Naive policy
         */
        template <typename ArgList, typename MetaList, typename Grid, enumtype::platform BackendId>
        struct prepare_temporaries_functor<ArgList, MetaList, Grid, BackendId, enumtype::Naive>
        {

            //TODO check the type of ArgList
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set<MetaList>::value, "wrong type for metadata");
            GRIDTOOLS_STATIC_ASSERT(is_grid<Grid>::value, "wrong type for grid");

            typedef MetaList metadata_set_t;

            /**
               @brief instantiate the \ref gridtools::domain_type for the temporary storages
            */
            struct instantiate_tmps
            {
                uint_t m_tile_i;// tile along i
                uint_t m_tile_j;// tile along j
                uint_t m_tile_k;// tile along k
                metadata_set_t& m_metadata_set;

                GT_FUNCTION
                instantiate_tmps( metadata_set_t& metadata_set_, uint_t tile_i, uint_t tile_j, uint_t tile_k
                    )
                    : m_metadata_set(metadata_set_)
                    , m_tile_i(tile_i)
                    , m_tile_j(tile_j)
                    , m_tile_k(tile_k)
                {}


                // ElemType: an element in the data field place-holders list
                template <typename ElemType>
                void operator()(pointer<ElemType>&  e) const {

                    GRIDTOOLS_STATIC_ASSERT(is_storage<ElemType>::value, "wrong type");
                    GRIDTOOLS_STATIC_ASSERT(ElemType::is_temporary, "wrong type (not temporary)");
                    GRIDTOOLS_STATIC_ASSERT(is_meta_storage<typename ElemType::storage_info_type>::value, "wrong metadata type");

                    if( !m_metadata_set.template present<pointer<
                        typename ElemType::storage_info_type const> >() )
                        //creates a metadata on the heap, passing on ownership
                        m_metadata_set.insert( pointer<typename ElemType::storage_info_type const>(
                                                   new typename ElemType::storage_info_type(m_tile_i
                                                                                      , m_tile_j
                                                                                      , m_tile_k)));

                    //ElemType::info_string.c_str();
                    //calls the constructor of the storage
                    e = new ElemType(
                        *m_metadata_set.template get<pointer<const typename ElemType::storage_info_type> >()
                        , "default tmp storage"
                        );
                }
            };

            static void prepare_temporaries(ArgList & arg_list, metadata_set_t& metadata_, Grid const& grid) {

#ifdef VERBOSE
                std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

                typedef boost::fusion::filter_view<ArgList,
                    is_temporary_storage<boost::mpl::_1> > view_type;

                view_type fview(arg_list);

                boost::fusion::for_each(fview,
                                        instantiate_tmps( metadata_,
                                                          grid.direction_i().total_length(),
                                                          grid.direction_j().total_length(),
                                                          grid.value_at_top()-grid.value_at_bottom()+1
                                            ));

            }

        };

        /**
           Specialization for Block policy
         */
        template <typename ArgList, typename MetaList, typename Grid, enumtype::platform BackendId>
        struct prepare_temporaries_functor
        <ArgList, MetaList, Grid,  BackendId, enumtype::Block >
        {

            //TODO implement a check for the ArgList type
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set<MetaList>::value, "wrong type for metadata");
            GRIDTOOLS_STATIC_ASSERT(is_grid<Grid>::value, "wrong type for Grid");

            typedef backend<BackendId, enumtype/*::strategy*/::Block> backend_type;
            /**
               @brief instantiate the \ref gridtools::domain_type for the temporary storages
            */
            struct instantiate_tmps
            {
                typedef MetaList metadata_set_t;
                metadata_set_t& m_metadata_set;
                uint_t m_offset_i;// offset along i
                uint_t m_offset_j;// offset along j
                uint_t m_offset_k;// offset along k
                uint_t m_n_i_threads;
                uint_t m_n_j_threads;

                GT_FUNCTION
                instantiate_tmps(metadata_set_t& metadata_set_,
                                 uint_t offset_i,
                                 uint_t offset_j,
                                 uint_t offset_k,
                                 uint_t m_n_i_threads,
                                 uint_t m_n_j_threads)
                    : m_metadata_set(metadata_set_)
                    , m_offset_i(offset_i)
                    , m_offset_j(offset_j)
                    , m_offset_k(offset_k)
                    , m_n_i_threads(m_n_i_threads)
                    , m_n_j_threads(m_n_j_threads)
                {}

                // ElemType: an element in the data field place-holders list
                template <typename ElemType>
                void operator()(pointer<ElemType>&  e) const {
                    GRIDTOOLS_STATIC_ASSERT(is_storage<ElemType>::value, "wrong type (not temporary)");
                    GRIDTOOLS_STATIC_ASSERT(ElemType::is_temporary, "wrong type (not temporary)");
                    GRIDTOOLS_STATIC_ASSERT(is_meta_storage<typename ElemType::storage_info_type>::value, "wrong metadata type");

                    typedef typename ElemType::storage_info_type meta_t;

                    //insert new type in the map only if not present already
                    if( !m_metadata_set.template present<pointer<const meta_t> >() )
                        //creates a metadata on the heap, passing on ownership
                        m_metadata_set.insert(pointer<meta_t const>(
                                                  new meta_t(
                                                  m_offset_i
                                                  , m_offset_j
                                                  , m_offset_k
                                                  , m_n_i_threads
                                                  , m_n_j_threads
                                                      ) )
                            );

                    //calls the constructor of the storage
                    e = new ElemType(
                        *m_metadata_set.template get<pointer<const meta_t> >()
                        , "blocked tmp storage");
                }
            };

            static void prepare_temporaries(ArgList & arg_list, MetaList & metadata_, Grid const& grid) {
                //static const enumtype::strategy StrategyType = Block;

#ifdef VERBOSE
                std::cout << "Prepare ARGUMENTS" << std::endl;
#endif
                typedef boost::fusion::filter_view<ArgList,
                    is_temporary_storage<boost::mpl::_1> > view_type;

                view_type fview(arg_list);
                boost::fusion::for_each(fview,
                                        instantiate_tmps
                                        ( metadata_,
                                          grid.i_low_bound(),
                                          grid.j_low_bound(),
                                          grid.value_at_top()-grid.value_at_bottom()+1,
                                          backend_type::n_i_pes()(grid.i_high_bound() - grid.i_low_bound()),
                                          backend_type::n_j_pes()(grid.j_high_bound() - grid.j_low_bound())
                                            )
                    );
            }

        };

        struct delete_tmps {
            template <typename Elem>
            GT_FUNCTION
            void operator()(Elem & elem) const {
                delete_pointer d;
                d(elem);
            }
        };

    } // namespace _impl

} // namespace gridtools
