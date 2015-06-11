using uint_t = unsigned int;
#include <iostream>
#include "grid.hpp"
#include <storage/base_storage.h>
#include <common/layout_map.h>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/adapted/mpl/detail/size_impl.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <type_traits>


template <int I, typename LocationType>
struct arg {
    using location_type = LocationType;
};

template <int I, typename T>
std::ostream& operator<<(std::ostream& s, arg<I,T>) {
    return s << "placeholder<" << I << ", " << T() << ">";
}

template <int I>
struct accessor {
    static const int value = I;
};

template <typename Functor,
          typename Grid,
          typename LocationType,
          typename PlcHldrs>
struct esf
{
    using functor = Functor;
    using grid = Grid;
    using location_type = LocationType;
    using plcs = PlcHldrs;
};


template <typename Functor,
          typename Grid,
          typename LocationType,
          typename ...PlcHldr0>
esf<Functor, Grid, LocationType, boost::mpl::vector<PlcHldr0...> >
make_esf(PlcHldr0... args) {
    return esf<Functor, Grid, LocationType, boost::mpl::vector<PlcHldr0...>>();
}

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


namespace gridtools {
    template <typename T, size_t D>
    std::ostream& operator<<(std::ostream &s, array<T,D> const& a) {
        s << " {  ";
        for (int i = 0; i < D-1; ++i) {
            s << a[i] << ", ";
        }
        s << a[D-1] << "  } ";

        return s;
    }

}

template <typename T, size_t D>
bool operator==(gridtools::array<T,D> const& a, gridtools::array<T,D> const& b) {
    gridtools::array<T,D> a0 = a;
    gridtools::array<T,D> b0 = b;
    std::sort(a0.begin(), a0.end());
    std::sort(b0.begin(), b0.end());
    return std::equal(a0.begin(), a0.end(), b0.begin());
}

struct on_cells_f {
    typedef accessor<0> out;
    typedef accessor<1> in;
    typedef accessor<2> out_edges_NOT_USED;
    typedef accessor<3> in_edges;

    template <typename GridAccessors>
    void
    operator()(GridAccessors /*const*/& eval/*, region*/) const {
        // std::cout << "i = " << eval.i()
        //           << " j = " << eval.j()
        //           << std::endl;
        auto ff = [](const double _in, const double _res) -> double {return _in+_res;};
        eval(out()) = eval(on_cells<in>(ff), 0.0) + eval(on_edges<in_edges>(ff), 0.0);
    }
};

#define EVAL(f,x,y)                                                     \
    std::cout << #f << ": " << gridtools::array<decltype(x),2>{x,y} << " -> " << (grid.f({x,y})) << std::endl

#define EVAL_C(f,x,y, result)                                                    \
    std::cout << #f << ": " << gridtools::array<decltype(x),2>{x,y} << " -> " << (grid.f({x,y})) << ", expect " << result; \
    std::cout << ": Passed? " << std::boolalpha << (grid.f({x,y}) == result) << std::endl

using gridtools::layout_map;
using gridtools::wrap_pointer;

int main() {
    using cell_storage_type = gridtools::base_storage<wrap_pointer<double>, layout_map<0,1,2> >;
    using edge_storage_type = gridtools::base_storage<wrap_pointer<double>, layout_map<0,1,2> >;

    using trapezoid_2D = gridtools::trapezoid_2D_no_tile<cell_storage_type, edge_storage_type>;

    uint_t NC = trapezoid_2D::u_cell_size_i(6);
    uint_t MC = trapezoid_2D::u_cell_size_j(12);
   
    uint_t NE = trapezoid_2D::u_edge_size_i(6);
    uint_t ME = trapezoid_2D::u_edge_size_j(12);

    std::cout << "NC = " << NC << " "
              << "MC = " << MC
              << std::endl;
    
    std::cout << "NE = " << NE << " "
              << "ME = " << ME
              << std::endl;

    cell_storage_type cells(trapezoid_2D::u_cell_size(gridtools::array<uint_t, 2>{NC, MC}));
    edge_storage_type edges(trapezoid_2D::u_edge_size(gridtools::array<uint_t, 2>{NE, ME}));

    trapezoid_2D grid(/*cells, edges,*/ 6, 12);

    cells.info();
    
    EVAL_C(cell2cells_ll_p0, 1, 1, (gridtools::array<uint_t,3>{9, 24, 25}));
    EVAL_C(cell2cells_ll_p0, 1, 2, (gridtools::array<uint_t,3>{10, 25, 26}));
    EVAL_C(cell2cells_ll_p1, 1, 3, (gridtools::array<uint_t,3>{19, 20, 35}));
    EVAL_C(cell2cells_ll_p1, 1, 4, (gridtools::array<uint_t,3>{20, 21, 36}));
    EVAL_C(cell2cells, 2, 3, (gridtools::array<uint_t,3>{33, 34, 49}));
    EVAL_C(cell2cells, 2, 4, (gridtools::array<uint_t,3>{26, 41, 42}));
    EVAL_C(cell2cells, 3, 3, (gridtools::array<uint_t,3>{49, 50, 65}));
    EVAL_C(cell2cells, 3, 4, (gridtools::array<uint_t,3>{42, 57, 58}));

    EVAL_C(edge2edges_ll_p0, 2, 3, (gridtools::array<uint_t,4>{66,67,59,82}));
    EVAL_C(edge2edges_ll_p1, 2, 3, (gridtools::array<uint_t,4>{43,28,51,67}));
    EVAL_C(edge2edges_ll_p2, 2, 3, (gridtools::array<uint_t,4>{51,59,52,83}));
    EVAL_C(edge2edges, 2, 2, (gridtools::array<uint_t,4>{48,56,49,80}));
    EVAL_C(edge2edges, 2, 3, (gridtools::array<uint_t,4>{64,57,65,80}));
    EVAL_C(edge2edges, 2, 4, (gridtools::array<uint_t,4>{41,26,49,65}));

    EVAL_C(cell2edges_ll_p0, 2, 3, (gridtools::array<uint_t,3>{51,59,67}));
    EVAL_C(cell2edges_ll_p1, 2, 3, (gridtools::array<uint_t,3>{67,52,83}));
    EVAL_C(cell2edges, 2, 3, (gridtools::array<uint_t,3>{65,50,81}));
    EVAL_C(cell2edges, 2, 4, (gridtools::array<uint_t,3>{58,50,66}));

    EVAL_C(edge2cells_ll_p0, 2, 3, (gridtools::array<uint_t,2>{42,35}));
    EVAL_C(edge2cells_ll_p1, 2, 3, (gridtools::array<uint_t,2>{27,35}));
    EVAL_C(edge2cells_ll_p2, 2, 3, (gridtools::array<uint_t,2>{35,43}));
    EVAL_C(edge2cells, 2, 3, (gridtools::array<uint_t,2>{33,40}));
    EVAL_C(edge2cells, 2, 4, (gridtools::array<uint_t,2>{25,33}));
    EVAL_C(edge2cells, 2, 5, (gridtools::array<uint_t,2>{33,41}));


    cell_storage_type cells_out(trapezoid_2D::u_cell_size(gridtools::array<uint_t, 2>{NC, MC}));
    edge_storage_type edges_out(trapezoid_2D::u_edge_size(gridtools::array<uint_t, 2>{NE, ME}));

    //    trapezoid_2D grid_out(cells_out, edges_out, NC, MC);

    typedef arg<0, trapezoid_2D::cells> in_cells;
    typedef arg<1, trapezoid_2D::cells> out_cells;
    typedef arg<2, trapezoid_2D::edges> out_edges;
    typedef arg<3, trapezoid_2D::edges> in_edges;

    auto x = make_esf<on_cells_f, trapezoid_2D, trapezoid_2D::cells>(in_cells(), out_cells(), out_edges(), in_edges());

    accessor_type<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>, trapezoid_2D> acc
        (boost::fusion::vector<cell_storage_type*, cell_storage_type*, edge_storage_type*, edge_storage_type*>(&cells_out, &cells, &edges_out, &edges), grid, 0,0);

    for (int i = 1; i < NC-1; ++i) {
        acc.set_ij(i, 1);
        for (int j = 2; j < MC-2; ++j) {
            acc.inc_j();
            decltype(x)::functor()(acc);
        }
    }

    // gridtools::computation * oncells =
    //     gridtools::make_computation<gridtools::backend<Host, Naive>
    //     (
    //      gridtools::make_mss
    //      (
    //       gridtools::execute<forward>(),
    //       gridtools::make_esf<on_cells_f>(in_grid_cells(), out_grid_cells())
    return 0;
}
