#pragma once

#include <common/array.hpp>
#include <cassert>
#include <boost/mpl/vector.hpp>
#include "virtual_storage.hpp"
#include "location_type.hpp"
#include "backend.hpp"
#include "array_addons.hpp"

namespace gridtools {

    extern char const cells_str[]="cells";
    extern char const edges_str[]="edges";
    extern char const vertexes_str[]="vertexes";

    namespace{
        using cells = location_type<0,2>;
        using edges = location_type<1,3>;
        using vertexes = location_type<2,1>;
    }

    template<typename T, typename ValueType=int_t>
    struct return_type{
        typedef array<ValueType, 0> type;
    };

    //static triple dispatch
    template <typename Location1>
    struct from{
        template <typename Location2>
        struct to{
            template <typename Color>
                struct with_color{

                template<typename ValueType>
                using return_t=typename return_type<from<Location1>::to<Location2>, ValueType >::type;

                template<typename Grid>
                static return_t<int_t> get(Grid const& grid_, array<int_t, 2> const& i){
                    //not supported
                    assert(false);
                }

                static return_t<array<uint_t, 3>> get_index(array<int_t, 2> const& i){
                    //not supported
                    assert(false);
                }
            };
        };
    };

    template <typename ValueType>
    struct return_type<from<cells>::template to<cells>, ValueType>{
        typedef array<ValueType, 3> type;
    };

    template <typename ValueType>
    struct return_type<from<cells>::template to<edges>, ValueType>{
        typedef array<ValueType, 3> type;
    };

    template <typename ValueType>
    struct return_type<from<cells>::template to<vertexes>, ValueType>{
        typedef array<ValueType, 3> type;
    };

    template <typename ValueType>
    struct return_type<from<edges>::template to<edges>, ValueType>{
        typedef array<ValueType, 4> type;
    };

    template <typename ValueType>
    struct return_type<from<edges>::template to<cells>, ValueType>{
        typedef array<ValueType, 2> type;
    };

    template <typename ValueType>
    struct return_type<from<edges>::template to<vertexes>, ValueType>{
        typedef array<ValueType, 4> type;
    };

    template <typename ValueType>
    struct return_type<from<vertexes>::template to<vertexes>, ValueType>{
        typedef array<ValueType, 6> type;
    };

    template <typename ValueType>
    struct return_type<from<vertexes>::template to<cells>, ValueType>{
        typedef array<ValueType, 6> type;
    };

    template <typename ValueType>
    struct return_type<from<vertexes>::template to<edges>, ValueType>{
        typedef array<ValueType, 6> type;
    };

    template<> template<> template<>
    struct from<cells>::to<cells>::with_color<static_int<1> >{

        template <typename ValueType>
        using return_t=typename return_type<from<cells>::to<cells>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<cells::value >(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                std::get<cells::value >(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                std::get<cells::value >(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1], i[2])};
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0], 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]+1, 0, i[1], i[2]}};
        }
    };

    template<> template<> template<>
    struct from<cells>::to<cells>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t=typename return_type<from<cells>::to<cells>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<0>(grid_.v_storage_tuple())._index(i[0], 1, i[1]-1, i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0], 1, i[1], i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0]-1, 1, i[1], i[2])};
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0], 1, i[1]-1, i[2]},
                { i[0], 1, i[1], i[2]},
                { i[0]-1, 1, i[1], i[2]}};//NOTE: different from above!!
        }
    };

    template<> template<> template<>
    struct from<vertexes>::to<vertexes>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t=typename return_type<from<vertexes>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                    std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1]-1, i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1]-1, i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1], i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0]-1, 0, i[1]+1, i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0]-1, 0, i[1], i[2])
            };
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0], 0, i[1]-1, i[2]},
                { i[0]+1, 0, i[1]-1, i[2]},
                { i[0]+1, 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]-1, 0, i[1]+1, i[2]},
                { i[0]-1, 0, i[1], i[2]},
            };
        }
    };

    template<> template<> template<>
    struct from<edges>::to<edges>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<edges>, ValueType>::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<1>(grid_.v_storage_tuple())._index(i[0], 1, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0]+1, 1, i[1]-1, i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 2, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 2, i[1]-1, i[2])};
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0], 1, i[1], i[2]},
                { i[0]+1, 1, i[1]-1, i[2]},
                { i[0], 2, i[1], i[2]},
                { i[0], 2, i[1]-1, i[2]}};
        }
    };


    template<> template<> template<>
    struct from<edges>::to<edges>::with_color<static_int<1> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<edges>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<1>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0]-1, 0, i[1]+1, i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 2, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0]-1, 2, i[1], i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0], 0, i[1], i[2]},
                { i[0]-1, 0, i[1]+1, i[2]},
                { i[0], 2, i[1], i[2]},
                { i[0]-1, 2, i[1], i[2]}};
        }
    };


    template<> template<> template<>
    struct from<edges>::to<edges>::with_color<static_int<2> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<edges>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<1>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 1, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0]+1, 1, i[1], i[2])};
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> > {
                { i[0], 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0], 1, i[1], i[2]},
                { i[0]+1, 1, i[1], i[2]}};//Different from above!!
        }
    };


    template<> template<> template<>
    struct from<cells>::to<edges>::with_color<static_int<1> >{

        template<typename ValueType=int_t>
        using return_t= typename return_type<from<cells>::to<edges>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<1>(grid_.v_storage_tuple())._index(i[0], 2, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0]+1, 1, i[1], i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 2, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]+1, 1, i[1], i[2]}};
        }
    };


    template<> template<> template<>
    struct from<cells>::to<edges>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t= typename return_type<from<cells>::to<edges>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<1>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 1, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 2, i[1], i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 0, i[1], i[2]},
                { i[0], 1, i[1], i[2]},
                { i[0], 2, i[1], i[2]}};
        }

    };

    template<> template<> template<>
    struct from<cells>::to<vertexes>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t= typename return_type<from<cells>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1], i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]+1, 0, i[1], i[2]}};
        }
    };

    template<> template<> template<>
    struct from<cells>::to<vertexes>::with_color<static_int<1> >{

        template <typename ValueType>
        using return_t= typename return_type<from<cells>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1], i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1]+1, i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0]+1, 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]+1, 0, i[1]+1, i[2]}};
        }
    };


    template<> template<> template<>
    struct from<edges>::to<cells>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<cells>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<0>(grid_.v_storage_tuple())._index(i[0], 1, i[1]-1, i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 1, i[1]-1, i[2]},
                { i[0], 0, i[1], i[2]}};
        }
    };


    template<> template<> template<>
    struct from<edges>::to<cells>::with_color<static_int<1> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<cells>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<0>(grid_.v_storage_tuple())._index(i[0]-1, 1, i[1], i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0]-1, 1, i[1], i[2]},
                { i[0], 0, i[1], i[2]}};
        }
    };

    template<> template<> template<>
    struct from<edges>::to<cells>::with_color<static_int<2> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<cells>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<0>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0], 1, i[1], i[2])};
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 0, i[1], i[2]},
                { i[0], 1, i[1], i[2]}};
        }
    };

    template<> template<> template<>
    struct from<edges>::to<vertexes>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1], i[2]),
                    std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1]-1, i[2])
            };
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 0, i[1], i[2]},
                { i[0], 1, i[1]+1, i[2]},
                { i[0]+1, 1, i[1], i[2]},
                { i[0]+1, 1, i[1]-1, i[2]},
            };
        }
    };

    template<> template<> template<>
    struct from<edges>::to<vertexes>::with_color<static_int<1> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0]-1, 0, i[1]+1, i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1], i[2])
            };
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 0, i[1], i[2]},
                { i[0]-1, 0, i[1]+1, i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]+1, 0, i[1], i[2]},
            };
        }
    };

    template<> template<> template<>
    struct from<edges>::to<vertexes>::with_color<static_int<2> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0], 0, i[1]+1, i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1], i[2]),
                std::get<2>(grid_.v_storage_tuple())._index(i[0]+1, 0, i[1]+1, i[2])
            };
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]+1, 0, i[1], i[2]},
                { i[0]+1, 0, i[1]-1, i[2]},
            };
        }
    };

    template<> template<> template<>
    struct from<vertexes>::to<cells>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t=typename return_type<from<vertexes>::to<cells>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                    std::get<0>(grid_.v_storage_tuple())._index(i[0]-1, 1, i[1]-1, i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0]-1, 0, i[1], i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0]-1, 1, i[1], i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0], 1, i[1]-1, i[2]),
                    std::get<0>(grid_.v_storage_tuple())._index(i[0], 0, i[1]-1, i[2])
            };
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0]-1, 1, i[1]-1, i[2]},
                { i[0]-1, 0, i[1], i[2]},
                { i[0]-1, 1, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0], 1, i[1]-1, i[2]},
                { i[0], 0, i[1]-1, i[2]},
            };
        }
    };

    template<> template<> template<>
    struct from<vertexes>::to<edges>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t=typename return_type<from<vertexes>::to<edges>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<int_t, 3> const& i){
            return return_t<int_t>{
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 1, i[1]-1, i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0]-1, 0, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0]-1, 2, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 1, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 0, i[1], i[2]),
                    std::get<1>(grid_.v_storage_tuple())._index(i[0], 2, i[1]-1, i[2])
            };
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0], 1, i[1]-1, i[2]},
                { i[0]-1, 0, i[1], i[2]},
                { i[0]-1, 2, i[1], i[2]},
                { i[0], 1, i[1], i[2]},
                { i[0], 0, i[1], i[2]},
                { i[0], 2, i[1]-1, i[2]},
            };
        }
    };

    /**
    */
    template <typename Backend>
    class trapezoid_2D_colored {
    public :

        using cells = location_type<0,2>;
        using edges = location_type<1,3>;
        using vertexes = location_type<2,1>;

        template <typename T>
        struct pointer_to;

        template <int I, uint_t D>
        struct pointer_to<location_type<I, D>> {
            using type = double*;
        };

        template <typename LocationType>
        using storage_t = typename Backend::template storage_type<LocationType>;

    private:
        template <typename LocationType>
        using v_storage_t = virtual_storage<typename storage_t<LocationType>::layout>;

        const gridtools::array<uint_t, 2> m_dims; // Sizes as cells in a multi-dimensional Cell array

        static constexpr int Dims = 2;

        std::tuple<v_storage_t<cells>, v_storage_t<edges>, v_storage_t<vertexes> > m_v_storage_tuple;

        using virtual_storage_types =
            typename boost::fusion::vector<v_storage_t<cells>*, v_storage_t<edges>*, v_storage_t<vertexes>*>;
        using storage_types = boost::mpl::vector<storage_t<cells>*, storage_t<edges>*, storage_t<vertexes>* >;
        virtual_storage_types m_virtual_storages;
    public:

        template <typename LocationType>
        uint_t size(LocationType location){return std::get</*storage_t<LocationType> cxx14*/ LocationType::value >(m_v_storage_tuple).size();}

        template <typename T>
        struct virtual_storage_type;

        template <int I, int D>
        struct virtual_storage_type<location_type<I, D> > {
            using type = typename boost::fusion::result_of::at_c<virtual_storage_types, I>::type;
        };

        template <typename T>
        struct storage_type;

        template <int I, ushort_t D>
        struct storage_type<location_type<I, D> > {
            using type = typename boost::mpl::at_c<storage_types, I>::type;
        };

        //specific for triangular cells
        static constexpr uint_t u_size_j(cells, int _M) {return _M+4;}
        static constexpr uint_t u_size_i(cells, int _N) {return _N+2;}
        static constexpr uint_t u_size_j(edges, int _M) {return 3*(_M/2)+6;}
        static constexpr uint_t u_size_i(edges, int _N) {return _N+2;}
        static constexpr uint_t u_size_j(vertexes, int _M) {return _M/2+3;}
        static constexpr uint_t u_size_i(vertexes, int _N) {return _N+3;}

        std::tuple<v_storage_t<cells>, v_storage_t<edges>, v_storage_t<vertexes> > const& v_storage_tuple() const {return m_v_storage_tuple;}

        trapezoid_2D_colored() = delete;
    public :

        template<typename ... UInt>
        trapezoid_2D_colored(uint_t first_, uint_t second_, UInt ... dims)
            : m_dims{second_, first_}
            , m_v_storage_tuple(// std::make_tuple(
                // v_storage_t<cells>(
                array<uint_t, v_storage_t<cells>::space_dimensions>
                {u_size_i(cells(), first_), cells::n_colors, u_size_j(cells() ,second_)/cells::n_colors, dims...}//)
                , // v_storage_t<edges>(
                array<uint_t, v_storage_t<edges>::space_dimensions>
                {u_size_i(edges(), first_), edges::n_colors, u_size_j(edges() , second_)/edges::n_colors, dims...}//))
                , // v_storage_t<vertexes>(
                array<uint_t, v_storage_t<vertexes>::space_dimensions>
                {u_size_i(vertexes(), first_), vertexes::n_colors, u_size_j(vertexes() , second_)/vertexes::n_colors, dims...}//))
             )
        {
            boost::fusion::at_c<cells::value>(m_virtual_storages) = &std::get<0>(m_v_storage_tuple);
            boost::fusion::at_c<edges::value>(m_virtual_storages) = &std::get<1>(m_v_storage_tuple);
            boost::fusion::at_c<vertexes::value>(m_virtual_storages) = &std::get<2>(m_v_storage_tuple);
        }

        virtual_storage_types const& virtual_storages() const {return m_virtual_storages;}

// #define DO_THE_MATH(stor, i,j,k)                \
//         m_v_ ## stor ## _storage._index(i,j,k)


        template <typename LocationType>
        array<int_t, 4> ll_indices(array<int_t, 3> const& i, LocationType) const {
            // std::cout << " *cells* " << std::endl;
            auto out = array<int_t, 4>{i[0], i[1]%static_cast<int_t>(LocationType::n_colors), i[1]/static_cast<int>(LocationType::n_colors), i[2]};
            std::cout << "Location: " << LocationType() << " "  << i << " -> " << out << std::endl;
            return array<int_t, 4>{i[0], i[1]%static_cast<int_t>(LocationType::n_colors), i[1]/static_cast<int>(LocationType::n_colors), i[2]};
        }

        template<typename LocationType>
        int_t ll_offset(array<uint_t, 4> const& i, LocationType) const {
#ifdef _GRID_H_DEBUG
            std::cout << " **";
            LocationType::print_name::apply();
            std::cout<<"offsets** "
                     << std::get</*storage_t<LocationType> cxx14*/ LocationType::value>(m_v_storage_tuple)._index(&i[0]) << " from ("
                      << i[0] << ", "
                      << i[1] << ", "
                      << i[2] << ", "
                      << i[3] << ")"
                      << std::endl;
#endif
            return std::get<LocationType::value>(m_v_storage_tuple)._index(&i[0]);
        }

        // methods returning the neighbors. Specializations according to the location type
        // needed a way to implement static double dispatch
        template<typename Location1, typename Location2, typename Color>
        typename return_type<typename from<Location1>::template to<Location2>>::type const
        ll_map( Location1, Location2, Color, array<int_t, 3> const& i) const{
            return from<Location1>::template to<Location2>::template with_color<Color>::get(*this, i);
        }


        // methods returning the neighbors. Specializations according to the location type
        // needed a way to implement static double dispatch
        template<typename Location1, typename Location2, typename Color>
        typename return_type<typename from<Location1>::template to<Location2>, array<uint_t, 4> >::type const
        ll_map_index( Location1, Location2, Color, array<uint_t, 3> const& i) const{
            return from<Location1>::template to<Location2>::template with_color<Color>::get_index(i);
        }

        template<typename Location2> // Works for cells or edges with same code
        typename return_type<typename from<cells>::template to<Location2>, array<uint_t, 4> >::type
        neighbors_indices_3(array<uint_t, 4> const& i, cells, Location2) const
        {
 #ifdef _GRID_H_DEBUG
            std::cout << "neighbors_indices_3 cells to " << Location2() << " "
                      << i[0] << ", " << i[1] << ", " << i[2] << ", " << i[3]
                      << std::endl;
#endif
            switch (i[1]%cells::n_colors) {
            case 0:
                return ll_map_index(cells(), Location2(), static_int<0>(), {i[0], i[2], i[3]});
                // return edge2edges_ll_p0_indices({i[0], i[2]});
            case 1:
                return ll_map_index(cells(), Location2(), static_int<1>(), {i[0], i[2], i[3]});
                // return edge2edges_ll_p1_indices({i[0], i[2]});
            }
        }

        template<typename Location2> // Works for cells or edges with same code
        typename return_type<typename from<edges>::template to<Location2>, array<uint_t, 4> >::type
        neighbors_indices_3(array<uint_t, 4> const& i, edges, Location2) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "neighbors_indices_3 edges edges "
                      << i[0] << ", " << i[1] << ", " << i[2]
                      << std::endl;
#endif
            switch (i[1]%edges::n_colors) {
            case 0:
                return ll_map_index(edges(), Location2(), static_int<0>(), {i[0], i[2], i[3]});
                // return edge2edges_ll_p0_indices({i[0], i[2]});
            case 1:
                return ll_map_index(edges(), Location2(), static_int<1>(), {i[0], i[2], i[3]});
                // return edge2edges_ll_p1_indices({i[0], i[2]});
            case 2:
                return ll_map_index(edges(), Location2(), static_int<2>(), {i[0], i[2], i[3]});
                // return edge2edges_ll_p2_indices({i[0], i[2]});
            }
        }


        template<typename Location2> // Works for cells or edges with same code
        typename return_type<typename from<vertexes>::template to<Location2>, array<uint_t, 4> >::type
        neighbors_indices_3(array<uint_t, 4> const& i, vertexes, Location2) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "neighbors_indices_3 vertexes to " << Location2() << " "
                      << i[0] << ", " << i[1] << ", " << i[2] << ", " << i[3]
                      << std::endl;
#endif
            return ll_map_index(vertexes(), Location2(), static_int<0>(), {i[0], i[2], i[3]});
        }

    };

}
