#pragma once

template <int I>
struct accessor {
    static const int value = I;
};

template <typename Accessor, typename Lambda>
struct on_cells_impl {
    using function = Lambda;
    function ff;
    on_cells_impl(function l)
        : ff(l)
    {}
};

template <typename Accessor, typename Lambda>
on_cells_impl<Accessor, Lambda>
on_cells(Lambda l) {
    return on_cells_impl<Accessor, Lambda>(l);
}
    
template <typename Accessor, typename Lambda>
struct on_edges_impl {
    using function = Lambda;
    function ff;
    on_edges_impl(function l)
        : ff(l)
    {}
};

template <typename Accessor, typename Lambda>
on_edges_impl<Accessor, Lambda>
on_edges(Lambda l) {
    return on_edges_impl<Accessor, Lambda>(l);
}
    
template <typename PlcVector, typename GridType>
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

    using storage_types = typename boost::fusion::result_of::as_vector<mpl_storage_types>::type; //typename std::remove_const<typename boost::fusion::result_of::as_vector<typename GridType::storage_types>::type>::type;
    
    using mpl_pointers_t_ = typename boost::mpl::transform<PlcVector,
                                                           get_pointer<GridType>
                                                           >::type;

    using pointers_t = typename boost::fusion::result_of::as_vector<mpl_pointers_t_>::type;

    using grid_type = GridType;
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
    double operator()(on_cells_impl<Arg, Accumulator> oncells, double initial = 0.0) const {
        auto neighbors = grid.cell2cells(/**(boost::fusion::at_c<Arg::value>(storages)),*/ {m_pos_i, m_pos_j});
        double result = initial;
        //std::cout << "on_cells " << Arg::value << std::endl;

        for (int i = 0; i<neighbors.size(); ++i) {
            result = oncells.ff(*(boost::fusion::at_c<Arg::value>(pointers)), result);
        }
    }

    template <typename Arg, typename Accumulator>
    double operator()(on_edges_impl<Arg, Accumulator> onedges, double initial = 0.0) const {
        auto neighbors = grid.cell2edges(/**(boost::fusion::at_c<Arg::value>(storages)),*/ {m_pos_i, m_pos_j});
        double result = initial;
        //std::cout << "on_edges " << Arg::value << std::endl;

        for (int i = 0; i<neighbors.size(); ++i) {
            result = onedges.ff(*(boost::fusion::at_c<Arg::value>(pointers)), result);
        }
    }

    template <int I>
    double& operator()(accessor<I> const& arg) const {
        return *(boost::fusion::at_c<I>(pointers));
    }
};
