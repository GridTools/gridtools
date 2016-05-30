#pragma once
#include <tuple>
#include "../../common/generic_metafunctions/accumulate.hpp"

namespace gridtools{
    namespace _impl{
        template<typename Layout, typename Plus, typename Minus, typename Tiles, typename Storage>
        struct compute_meta_storage;

        template <typename Layout, typename P1, typename P2, typename ... Plus, typename M1, typename M2, typename ... Minus, typename T1, typename T2, typename ... Tiles, typename Storage>
        struct compute_meta_storage<Layout, std::tuple<P1, P2, Plus...>, std::tuple<M1, M2, Minus ...>, std::tuple<T1, T2, Tiles ...>, Storage >{

            typedef typename boost::mpl::if_c<(Storage::field_dimensions>1),
                meta_storage_cache<Layout,
                P1::value-M1::value+T1::value, P2::value-M2::value+T2::value, //first 2 dimensions are special (the block)
                ( (Plus::value - Minus::value) >0 ? (Tiles::value - Minus::value + Plus::value) : 1) ...
                , Storage::field_dimensions>,
                meta_storage_cache<Layout,
                P1::value-M1::value+T1::value, P2::value-M2::value+T2::value, //first 2 dimensions are special (the block)
                ( (Plus::value - Minus::value) >0 ? (Tiles::value - Minus::value + Plus::value) : 1) ...
                > >::type type;
        };

        template <typename T>
        struct generate_layout_map;

        template <uint_t ... Id>
        struct generate_layout_map<gt_integer_sequence<uint_t, Id ...> >{
            typedef layout_map<Id ...> type;
        };

        template<typename Minus, typename Plus, typename Tiles >
        struct compute_size;

#ifndef CUDA8
        template<typename ... Minus, typename ... Plus, typename ... Tiles >
        struct compute_size<std::tuple<Minus ...>, std::tuple<Plus ...>, std::tuple<Tiles ...> >{
            static constexpr auto value=accumulate(multiplies(), (Plus::value+Tiles::value-Minus::value) ...);
        };
#endif

    }//namespace _impl
}//namespace gridtools
