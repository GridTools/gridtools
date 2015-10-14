#pragma once

#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/sequence/intrinsic/at.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/as_set.hpp>

#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/container/vector/vector_fwd.hpp>
#include <boost/fusion/include/vector_fwd.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/sequence/io.hpp>
#include <boost/fusion/include/io.hpp>

#include <common/array.hpp>
#include <cassert>
#include <boost/mpl/vector.hpp>
#include "location_type.hpp"
#include "array_addons.hpp"

namespace gridtools {

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
                static return_t<int_t> get(Grid const& grid_, array<uint_t, 2> const& i){
                    //not supported
                    assert(false);
                }

                static return_t<array<uint_t, 3>> get_index(array<uint_t, 2> const& i){
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[2])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[2])
            };
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0], 1, i[1]-1, i[2]},
                { i[0], 1, i[1], i[2]},
                { i[0]-1, 1, i[1], i[2]}
            };
        }
    };

    template<> template<> template<>
    struct from<vertexes>::to<vertexes>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t=typename return_type<from<vertexes>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[3]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[4]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[5])
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[3])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[3])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[3])
            };
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> > {
                { i[0], 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0], 1, i[1], i[2]},
                { i[0]+1, 1, i[1], i[2]}};
        }
    };


    template<> template<> template<>
    struct from<cells>::to<edges>::with_color<static_int<1> >{

        template<typename ValueType=int_t>
        using return_t= typename return_type<from<cells>::to<edges>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[2])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[2])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[2])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[2])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[1])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[1])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[1])
            };
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[3])
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
    struct from<edges>::to<vertexes>::with_color<static_int<1> >{

        template <typename ValueType>
        using return_t= typename return_type<from<edges>::to<vertexes>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[3])
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<vertexes::value>(grid_.virtual_storages()).index(get_index(i)[3])
            };
        }

        static return_t<array<uint_t, 4>> get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4>>{
                { i[0], 0, i[1], i[2]},
                { i[0], 0, i[1]+1, i[2]},
                { i[0]+1, 0, i[1], i[2]},
                { i[0]+1, 0, i[1]+1, i[2]},
            };
        }
    };

    template<> template<> template<>
    struct from<vertexes>::to<cells>::with_color<static_int<0> >{

        template <typename ValueType>
        using return_t=typename return_type<from<vertexes>::to<cells>, ValueType >::type;

        template<typename Grid>
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[3]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[4]),
                boost::fusion::at_c<cells::value>(grid_.virtual_storages()).index(get_index(i)[5])
            };
        }

        static return_t<array<uint_t, 4> > get_index(array<uint_t, 3> const& i){
            return return_t<array<uint_t, 4> >{
                { i[0]-1, 1, i[1]-1, i[2]},
                { i[0]-1, 0, i[1], i[2]},
                { i[0]-1, 1, i[1], i[2]},
                { i[0], 0, i[1], i[2]},
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
        static return_t<int_t> get(Grid const& grid_, array<uint_t, 3> const& i){
            return return_t<int_t>{
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[0]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[1]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[2]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[3]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[4]),
                boost::fusion::at_c<edges::value>(grid_.virtual_storages()).index(get_index(i)[5])
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

        //    private:
        template <typename LocationType>
        using v_storage_t = typename Backend::template meta_storage_t<LocationType>;

        template <typename LocationType, typename ValueType>
        using storage_t = typename Backend::template storage_t<LocationType, ValueType>;

        const gridtools::array<uint_t, 2> m_dims; // Sizes as cells in a multi-dimensional Cell array

        static constexpr int Dims = 2;
        static constexpr int n_colors = 2;

        using virtual_storage_types =
            boost::fusion::vector3<
            v_storage_t<cells>, v_storage_t<edges>, v_storage_t<vertexes>>;

        using storage_types = boost::mpl::vector<storage_t<cells, double>*,
                                                 storage_t<edges, double>*,
                                                 storage_t<vertexes, double>* >;

        virtual_storage_types m_virtual_storages;
    public:

        using n_locations = static_uint<boost::mpl::size<storage_types>::value >;
        template <typename LocationType>
        uint_t size(LocationType location){return boost::fusion::at_c<LocationType::value >(m_virtual_storages).size();}

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

        trapezoid_2D_colored() = delete;
    public :

        template<typename ... UInt>
        trapezoid_2D_colored(uint_t first_, uint_t second_, UInt ... dims)
            : m_dims{second_, first_},
             m_virtual_storages(
                 v_storage_t<cells>(array<uint_t, v_storage_t<cells>::space_dimensions>{first_, cells::n_colors, second_, dims...}),
                 v_storage_t<edges>(array<uint_t, v_storage_t<edges>::space_dimensions>{first_, edges::n_colors, second_, dims...}),
                 //here we assume by convention that the dual grid (vertexes) have one more grid point
                 v_storage_t<vertexes>(array<uint_t, v_storage_t<vertexes>::space_dimensions>{first_, vertexes::n_colors,  second_+1, dims...})
             )
        {}

        virtual_storage_types const& virtual_storages() const {return m_virtual_storages;}

        template <typename LocationType>
        storage_t<LocationType, double> make_storage(char const* name) const {
            return storage_t<LocationType, double>(boost::fusion::at_c<LocationType::value>
                                                   (m_virtual_storages), name);
        }

        template <typename LocationType>
        array<int_t, 4> ll_indices(array<int_t, 3> const& i, LocationType) const {
            // std::cout << " *cells* " << std::endl;
            auto out = array<int_t, 4>{i[0], i[1]%static_cast<int_t>(LocationType::n_colors), i[1]/static_cast<int>(LocationType::n_colors), i[2]};
            return array<int_t, 4>{i[0], i[1]%static_cast<int_t>(LocationType::n_colors), i[1]/static_cast<int>(LocationType::n_colors), i[2]};
        }

        template<typename LocationType>
        int_t ll_offset(array<uint_t, 4> const& i, LocationType) const {
#ifdef _GRID_H_DEBUG
            std::cout << " **";
            LocationType::print_name::apply();
            std::cout<<"offsets** "
                     << boost::fusion::at_c<LocationType::value>(m_virtual_storages).index(&i[0]) << " from ("
                      << i[0] << ", "
                      << i[1] << ", "
                      << i[2] << ", "
                      << i[3] << ")"
                      << std::endl;
#endif
            return boost::fusion::at_c<LocationType::value>(m_virtual_storages).index(i);
        }

        // methods returning the neighbors. Specializations according to the location type
        // needed a way to implement static double dispatch
        template<typename Location1, typename Location2, typename Color>
        typename return_type<typename from<Location1>::template to<Location2>>::type const
        ll_map( Location1, Location2, Color, array<uint_t, 3> const& i) const{
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

    template<typename T> struct is_grid : boost::mpl::false_{};

    template <typename Backend>
    struct is_grid<trapezoid_2D_colored<Backend> > : boost::mpl::true_ {};

} // namespace gridtools
