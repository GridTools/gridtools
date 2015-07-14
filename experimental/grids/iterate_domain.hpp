#pragma once
#include "location_type.hpp"
#include <type_traits>

#define _ACCESSOR_H_DEBUG_

// template <typename F1, typename F2>
// struct make_mapreduce {
//     using map_type = F1;
//     using reduce_type = F2;
// };


/**
   Map function that uses compile time (stateless) accessors to be
   evaluated later. Another version would have the Arguments to be
   a fusion vector (for instance), so that each argument can carry
   additional state, like a constant value.
 */
template <typename MapF, typename LocationType, typename ... Arguments>
struct map_function {
    using location_type = LocationType;
    using argument_types = std::tuple<Arguments...>;
    using function_type = MapF;

    const function_type m_function;
    argument_types m_arguments;

    map_function(function_type f, Arguments... args)
        : m_function(f)
        , m_arguments(args...)
    {}

    template <int I>
    typename std::tuple_element<I, argument_types>::type const&
    argument() const {
        return std::get<I>(m_arguments);
    }

    location_type location() const {return location_type();}

    function_type function() const {return m_function;}
};

/**
initial version of this that should check if all args have the same location type
*/
template <typename Arg0, typename ... Args>
struct get_location_type_of {
    using type = typename Arg0::location_type;
};

template <typename MapF, typename ... Args>
map_function<MapF, typename get_location_type_of<Args...>::type, Args...>
map(MapF const& f, Args... args) {
    return map_function<MapF, typename get_location_type_of<Args...>::type, Args...>(f, args...);
}

template <typename T>
struct identity {
    T operator()(T v) const
    {
        return v;
    }
};

/**
   This struct is the one holding the function to apply when iterating
   on neighbors
 */
template <typename ValueType
          , typename DstLocationType
          , typename ReductionFunction
          , typename MapFunction
          >
class on_neighbors_impl {
    using map_function = MapFunction;
    using reduction_function = ReductionFunction;
    using dst_location_type = DstLocationType;
    using value_type = ValueType;

    const reduction_function m_reduction;
    const map_function m_map;
    const value_type m_value;

public:
    on_neighbors_impl(const reduction_function l, map_function a, value_type v)
        : m_reduction(l)
        , m_map(a)
        , m_value(v)
    {}

    value_type value() const {return m_value;}

    reduction_function reduction() const {return m_reduction;}

    map_function map() const {return m_map;}

    on_neighbors_impl(on_neighbors_impl const& other)
        : m_reduction(other.m_reduction)
        , m_map(other.m_map)
        , m_value(other.m_value)
    {}

    dst_location_type location() const
    {
        return dst_location_type();
    }
};


template <typename Reduction
          , typename ValueType
          , typename Map
          >
on_neighbors_impl<ValueType
                  , typename Map::location_type
                  , Reduction
                  , Map
                  >
reduce_on_something(Reduction function
                    , ValueType initial
                    , Map mapf)
{
    return on_neighbors_impl<ValueType
                             , typename Map::location_type
                             , Reduction
                             , Map
                             >(function, mapf, initial);
}

template <typename Reduction
          , typename ValueType
          , typename Map
          >
on_neighbors_impl<ValueType
                  , typename Map::location_type
                  , Reduction
                  , Map
                  >
reduce_on_edges(Reduction function
                , ValueType initial
                , Map mapf)
{
    static_assert(std::is_same<typename Map::location_type, gridtools::location_type<1>>::value,
                  "The map function (for a nested call) provided to 'on_edges' is not on edges");
    return reduce_on_something(function, initial, mapf);
}

template <typename Reduction
          , typename ValueType
          , typename Map
          >
on_neighbors_impl<ValueType
                  , typename Map::location_type
                  , Reduction
                  , Map
                  >
reduce_on_cells(Reduction function
                , ValueType initial
                , Map mapf)
{
    static_assert(std::is_same<typename Map::location_type, gridtools::location_type<0>>::value,
                  "The map function (for a nested call) provided to 'on_cellss' is not on cells");
    return reduce_on_something(function, initial, mapf);
}

/**
   This is the type of the accessors accessed by a stencil functor.
   It's a pretty minima implementation.
 */
template <int I, typename LocationType>
struct accessor {
    using this_type = accessor<I, LocationType>;
    using location_type = LocationType;
    static const int value = I;

    template <typename Reduction
              , typename ValueType
              , typename Map // Can be an accessor, too
              >
    static
    on_neighbors_impl<ValueType, gridtools::location_type<0>, Reduction, Map>
    reduce_on_cells(Reduction red, ValueType initial, Map map) {
    return on_neighbors_impl<ValueType
                             , gridtools::location_type<0>
                             , Reduction
                             , Map
                             >(red, map, initial);
    }

    location_type location() const {
        return location_type();
    }
};


/**
   This class is basically the iterate domain. It contains the
   ways to access data and the implementation of iterating on neighbors.
 */
template <typename PlcVector, typename GridType, typename LocationType>
struct iterate_domain {

private:

    template <typename GridType_>
    struct get_pointer {
        template <typename PlcType>
        struct apply {
            using type = typename GridType_::template pointer_to<typename PlcType::location_type>::type;
        };
    };
    template <typename GridType_>
    struct get_storage {
        template <typename PlcType>
        struct apply {
            using type = typename GridType_::template storage_type<typename PlcType::location_type>::type;
        };
    };

public:
    using mpl_storage_types = typename boost::mpl::transform<PlcVector,
                                                             get_storage<GridType>
                                                             >::type;

    using storage_types = typename boost::fusion::result_of::as_vector<mpl_storage_types>::type;

    using mpl_pointers_t_ = typename boost::mpl::transform<PlcVector,
                                                           get_pointer<GridType>
                                                           >::type;

    using pointers_t = typename boost::fusion::result_of::as_vector<mpl_pointers_t_>::type;

    using grid_type = GridType;
    using location_type = LocationType;
private:
    storage_types storages;
    pointers_t pointers;
    grid_type const& m_grid;

    int m_ll_i;
    int m_ll_j;
    int m_ll_k;

    template <typename PointersT, typename StoragesT>
    struct _set_pointers
    {
        PointersT &m_pt;
        StoragesT const &m_st;
        _set_pointers(PointersT& pt, StoragesT const &st): m_pt(pt), m_st(st) {}

        template <typename Index>
        void operator()(Index) {
            double * ptr = const_cast<double*>((boost::fusion::at_c<Index::value>(m_st))->min_addr());

            boost::fusion::at_c<Index::value>(m_pt) = ptr;
        }
    };

    template <typename LocationT, typename PointersT, typename StoragesT, typename GridT>
    struct _set_pointers_to
    {
        PointersT &m_pt;
        StoragesT const &m_st;
        GridT const& m_g;
        const int m_ll_i;
        const int m_ll_j;
        const int m_ll_k;

        _set_pointers_to(PointersT& pt,
                         StoragesT const &st,
                         GridT const& g,
                         const int ll_i,
                         const int ll_j,
                         const int ll_k)
            : m_pt(pt)
            , m_st(st)
            , m_g(g)
            , m_ll_i(ll_i)
            , m_ll_j(ll_j)
            , m_ll_k(ll_k)
        {}

        template <typename Index>
        void operator()(Index) {
            double * ptr = const_cast<double*>((boost::fusion::at_c<Index::value>(m_st))->min_addr())
                + (boost::fusion::at_c<LocationT::value>(m_g.virtual_storages())->_index(m_ll_i, m_ll_j, m_ll_k));

            boost::fusion::at_c<Index::value>(m_pt) = ptr;
        }
    };

    template <int Coordinate, typename PointersT, typename GridT>
    struct _move_pointers
    {
        PointersT &m_pt;
        GridT const &m_g;

        _move_pointers(PointersT& m_pt, GridT const& m_g): m_pt(m_pt), m_g(m_g) {}

        template <typename Index>
        void operator()(Index) {
            auto value = boost::fusion::at_c<boost::mpl::at_c<PlcVector, Index::value>::type::location_type::value>
                (m_g.virtual_storages())->template strides<Coordinate>();
            //std::cout << "Stide<" << Index::value << "> for coordinate " << Coordinate << " = " << value << std::endl;
            boost::fusion::at_c<Index::value>(m_pt) += value;
        }
    };

    void _increment_pointers_k()
    {
        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
        boost::mpl::for_each<indices>(_move_pointers<2, pointers_t, grid_type>(pointers, m_grid));
    }

    void _reset_pointers()
    {
        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
        boost::mpl::for_each<indices>(_set_pointers<pointers_t, storage_types>(pointers, storages));
    }



    template <typename LocationT>
    void _set_pointers_to_ll() {
        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
        boost::mpl::for_each<indices>(_set_pointers_to<LocationT, pointers_t, storage_types, grid_type>(pointers, storages, m_grid, m_ll_i, m_ll_j, m_ll_k));
    }

public:
    iterate_domain(storage_types const& storages, GridType const& m_grid)
        : storages(storages)
        , m_grid(m_grid)
    {
        _reset_pointers();
    }

    GridType const& grid() const {return m_grid;}

    void inc_ll_k() {++m_ll_k; _increment_pointers_k();}

    template <typename LocationT>
    void set_ll_ijk(int i, int j, int k) {
        m_ll_i = i;
        m_ll_j = j;
        m_ll_k = k;
        _set_pointers_to_ll<LocationT>();
    }

    template <typename ValueType
              , typename LocationTypeT
              , typename Reduction
              , typename MapF
              , typename ...Arg0
              >
    double operator()(on_neighbors_impl<ValueType, LocationTypeT, Reduction, map_function<MapF, LocationTypeT, Arg0...>> onneighbors) const {
        auto current_position = gridtools::array<uint_t, 3>(static_cast<uint_t>(m_ll_i),
                                                            static_cast<uint_t>(m_ll_j),
                                                            static_cast<uint_t>(m_ll_k));

        const auto neighbors = m_grid.neighbors_indices_3(current_position
                                                          , location_type()
                                                          , onneighbors.location() );
        std::cout << "Entry point " << current_position << " Neighbors: " << neighbors << std::endl;

        double result = onneighbors.value();

        for (int i = 0; i<neighbors.size(); ++i) {
            result = onneighbors.reduction()( _evaluate(onneighbors.map(), neighbors[i]), result );
        }

        return result;
    }

    template <typename ValueType
              , typename LocationTypeT
              , typename Reduction
              , int I
              , typename L
              >
    double operator()(on_neighbors_impl<ValueType, LocationTypeT, Reduction, accessor<I,L>> onneighbors) const {
        auto current_position = gridtools::array<uint_t, 3>(static_cast<uint_t>(m_ll_i),
                                                            static_cast<uint_t>(m_ll_j),
                                                            static_cast<uint_t>(m_ll_k));

        const auto neighbors = m_grid.neighbors_indices_3(current_position
                                                          , location_type()
                                                          , onneighbors.location() );
        std::cout << "Entry point " << current_position << " Neighbors: " << neighbors << std::endl;

        double result = onneighbors.value();

        for (int i = 0; i<neighbors.size(); ++i) {
            result = onneighbors.reduction()( _evaluate(onneighbors.map(), neighbors[i]), result );
        }

        return result;
    }

    template <int I, typename LT>
    double& operator()(accessor<I,LT> const& arg) const {
        //std::cout << "ADDR " << std::hex << (boost::fusion::at_c<I>(pointers)) << std::dec << std::endl;
        return *(boost::fusion::at_c<I>(pointers));
    }

private:

    template <int I, typename L, typename IndexArray>
    double _evaluate(accessor<I,L>, IndexArray const& position) const {
        std::cout << "_evaluate(accessor<I,L>...) " << L() << ", " << position << std::endl;
        int offset = m_grid.ll_offset(position, typename accessor<I,L>::location_type());
        return *(boost::fusion::at_c<accessor<I,L>::value>(pointers)+offset);
    }

    template <typename MapF, typename LT, typename Arg0, typename IndexArray>
    double _evaluate(map_function<MapF, LT, Arg0> map, IndexArray const& position) const {
        std::cout << "_evaluate(map_function<MapF, LT, Arg0>...) " << LT() << ", " << position << std::endl;
        int offset = m_grid.ll_offset(position, map.location());
        return map.function()(_evaluate(map.template argument<0>(), position));
    }

    template <typename MapF, typename LT, typename Arg0, typename Arg1, typename IndexArray>
    double _evaluate(map_function<MapF, LT, Arg0, Arg1> map, IndexArray const& position) const {
        std::cout << "_evaluate(map_function<MapF, LT, Arg0, Arg1>...) " << LT() << ", " << position << std::endl;
        int offset = m_grid.ll_offset(position, map.location());
        return map.function()(_evaluate(map.template argument<0>(), position)
                              , _evaluate(map.template argument<1>(), position));
    }

    template <typename ValueType
              , typename LocationTypeT
              , typename Reduction
              , typename Map
              , typename IndexArray>
    double _evaluate(on_neighbors_impl<ValueType, LocationTypeT, Reduction, Map > onn, IndexArray const& position) const {
        const auto neighbors = m_grid.neighbors_indices_3(position
                                                          , onn.location()
                                                          , onn.location() );

        std::cout << "_evaluate(on_neighbors_impl<ValueType, ...) " << LocationTypeT() << ", " << position << " Neighbors: " << neighbors << std::endl;

        double result = onn.value();

        for (int i = 0; i<neighbors.size(); ++i) {
            result = onn.reduction()(_evaluate(onn.map(), neighbors[i]), result);
        }

        return result;
    }

};
