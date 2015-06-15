#pragma once
#include "location_type.hpp"
#include <type_traits>
/**
   This struct is the one holding the function to apply when iterating
   on neighbors
 */
template <typename Accessor, typename Lambda>
struct on_neighbors_impl {
    using accessor = Accessor;
    using function = Lambda;
    using location_type = typename Accessor::location_type;

    function ff;

    on_neighbors_impl(function l)
        : ff(l)
    {}
};

/**
   User friendly interface to let iterate on neighbors of a location of a location type
   (the one of the iteraion space). The neighbors are of a location type Accessor::location_type
 */
template <typename Accessor, typename Lambda>
on_neighbors_impl<Accessor, Lambda>
on_neighbors(Lambda l) {
    return on_neighbors_impl<Accessor, Lambda>(l);
}


template <typename NextStep>
struct nested_neighbors_impl {
    using next_type = NextStep;
    using location_type = typename next_type::location_type;
    using function_type = typename next_type::function_type;
    
    next_type m_next;
    
    nested_neighbors_impl(function_type ff)
        : m_next(ff)
        {}

    next_type next()
    {
        return m_next;
    }
};

template <typename Accessor, typename Lambda>
struct nested_neighbors_impl<on_neighbors_impl<Accessor, Lambda>> {
    using location_type = typename Accessor::location_type;
    using next_type = on_neighbors_impl<Accessor, Lambda>; 
    using function_type = Lambda;
   
    next_type m_next;
    
    nested_neighbors_impl(function_type ff)
        : m_next(ff)
        {}

    next_type next()
    {
        return m_next;
    }
};

/**
   User friendly interface to let iterate on neighbor cells of a cell
 */
template <typename Accessor, typename Lambda>
on_neighbors_impl<Accessor, Lambda>
on_cells(Accessor, Lambda l) {
    static_assert(std::is_same<typename Accessor::location_type, gridtools::location_type<0>>::value,
        "The accessor provided to 'on_cells' is not on cells");
    return on_neighbors_impl<Accessor, Lambda>(l);
}

/**
   User friendly interface to let iterate on neighbor cells of a cell
 */
template <typename Accessor, typename Lambda>
nested_neighbors_impl<on_neighbors_impl<Accessor, Lambda> >
on_cells(on_neighbors_impl<Accessor, Lambda> l) {
    // static_assert(std::is_same<typename Accessor::location_type, gridtools::location_type<0>>::value,
    //     "The accessor provided to 'on_cells' is not on cells");
    return nested_neighbors_impl<on_neighbors_impl<Accessor, Lambda>>(l.ff);
}

/**
   User friendly interface to let iterate on neighbor cells of a cell
 */
template <typename NextStep>
nested_neighbors_impl<NextStep>
on_cells(nested_neighbors_impl<NextStep> l) {
    // static_assert(std::is_same<typename Accessor::location_type, gridtools::location_type<0>>::value,
    //     "The accessor provided to 'on_cells' is not on cells");
    return nested_neighbors_impl<NextStep>(l.m_next);
};


/**
   User friendly interface to let iterate on neighbor edges of a edge
 */
template <typename Accessor, typename Lambda>
on_neighbors_impl<Accessor, Lambda>
on_edges(Accessor, Lambda l) {
    static_assert(std::is_same<typename Accessor::location_type, gridtools::location_type<1>>::value,
        "The accessor provided to 'on_edges' is not on edges");
    return on_neighbors_impl<Accessor, Lambda>(l);
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

    template <typename Lambda>
    on_neighbors_impl<this_type, Lambda>
    operator()(Lambda l) const {
        return on_neighbors_impl<this_type, Lambda>(l);
    }
    template <typename Lambda>
    static
    on_neighbors_impl<this_type, Lambda>
    neighbors(Lambda l) {
        return on_neighbors_impl<this_type, Lambda>(l);
    }
};


/**
   This class is basically the iterate domain. It contains the
   ways to access data and the implementation of iterating on neighbors.
 */
template <typename PlcVector, typename GridType, typename LocationType>
struct accessor_type {

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
    grid_type const& grid;
    uint_t m_pos_i;
    uint_t m_pos_j;

    template <typename PointersT, typename StoragesT>
    struct _set_pointers 
    {
        PointersT &pt;
        StoragesT const &st;
        _set_pointers(PointersT& pt, StoragesT const &st): pt(pt), st(st) {}

        template <typename Index>
        void operator()(Index) {
            double * ptr = const_cast<double*>((boost::fusion::at_c<Index::value>(st))->min_addr());
            std::cout << " -------------------------> Pointer " << std::hex << ptr << std::dec << std::endl;
            boost::fusion::at_c<Index::value>(pt) = ptr;
        }
    };
    
    template <int Coordinate, typename PointersT, typename GridT>
    struct _move_pointers 
    {
        PointersT &pt;
        GridT const &g;

        _move_pointers(PointersT& pt, GridT const& g): pt(pt), g(g) {}

        template <typename Index>
        void operator()(Index) {
            auto value = boost::fusion::at_c<boost::mpl::at_c<PlcVector, Index::value>::type::location_type::value>
                (g.virtual_storages())->template strides<Coordinate>();
            //std::cout << "Stide<" << Index::value << "> for coordinate " << Coordinate << " = " << value << std::endl;
            boost::fusion::at_c<Index::value>(pt) += value;
        }
    };
    
    void _increment_pointers_i()
    {
        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
        boost::mpl::for_each<indices>(_move_pointers<0, pointers_t, grid_type>(pointers, grid));
    }

    void _increment_pointers_j() 
    {
        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
        boost::mpl::for_each<indices>(_move_pointers<1, pointers_t, grid_type>(pointers, grid));
    }

    void _reset_pointers() 
    {
        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
        boost::mpl::for_each<indices>(_set_pointers<pointers_t, storage_types>(pointers, storages));
    }

    struct printplc {
        template <typename T>
        void operator()(T const& e) const {
            std::cout << "printing placeholders: " << e << std::endl;
        }
    };

public:
    accessor_type(storage_types const& storages, GridType const& grid, uint_t pos_i, uint_t pos_j)
        : storages(storages)
        , grid(grid)
        , m_pos_i(pos_i)
        , m_pos_j(pos_j)
    {
        _reset_pointers();
        boost::mpl::for_each<PlcVector>(printplc());
    }

    void inc_i() {++m_pos_i; _increment_pointers_i();}
    void inc_j() {++m_pos_j; _increment_pointers_j();}
    void reset_i() {m_pos_i = 0;}
    void reset_j() {m_pos_j = 0;}
    void set_ij(int i, int j) {
        m_pos_i = i;
        m_pos_j = j;
    }
    uint_t i() const {return m_pos_i;}
    uint_t j() const {return m_pos_j;}

    template <typename Arg, typename Accumulator>
    double operator()(on_neighbors_impl<Arg, Accumulator> onneighbors, double initial = 0.0) const {

        auto neighbors = grid.neighbors( {m_pos_i, m_pos_j}, location_type(), typename Arg::location_type() );
        double result = initial;
        //std::cout << "on_cells " << Arg::value << std::endl;

        for (int i = 0; i<neighbors.size(); ++i) {
            result = onneighbors.ff(*(boost::fusion::at_c<Arg::value>(pointers)+neighbors[i]), result);
        }
    }

    template <typename Defer>
    double operator()(nested_neighbors_impl<Defer> incomplete, double initial = 0.0) const {
        
        auto neighbors = grid.ll_indices( {m_pos_i, m_pos_j}, location_type() );
        incloop_over_neighbors<location_type>(incomplete, initial, neighbors);
    }


    template <int I, typename LT>
    double& operator()(accessor<I,LT> const& arg) const {
        return *(boost::fusion::at_c<I>(pointers));
    }

private:

    template <typename LocationTypeSrc, typename Defer>
    double incloop_over_neighbors(nested_neighbors_impl<Defer> __onneighbors,
                             double initial,
                             gridtools::array<uint_t, 3> const& indices) const
    {
        auto neighbors = grid.neighbors_indices_3( indices, LocationTypeSrc(), typename Defer::location_type() );
        for (int i = 0; i<neighbors.size(); ++i) {
            initial = incloop_over_neighbors<typename Defer::location_type>(__onneighbors.next(), initial, neighbors[i]);
        }
    }

    template <typename LocationTypeSrc, typename Arg, typename Accumulator>
    double incloop_over_neighbors(on_neighbors_impl<Arg, Accumulator> onneighbors,
                             double initial,
                             gridtools::array<uint_t, 3> const& indices) const
    {
        auto neighbors = grid.neighbors_indices_3( indices, LocationTypeSrc(), typename Arg::location_type() );
        for (int i = 0; i<neighbors.size(); ++i) {
            initial = onneighbors.ff(*(boost::fusion::at_c<Arg::value>(pointers)+grid.ll_offset(neighbors[i], typename Arg::location_type())), initial);
        }
    }
};
