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

template <int I>
struct accessor {
    static const int value = I;
};

template <typename Functor,
          typename Grid,
          typename PlcHldrs>
struct esf
{
    using functor = Functor;
    using grid = Grid;
    using plcs = PlcHldrs;
};


template <typename Functor,
          typename Grid,
          typename PlcHldr0,
          typename PlcHldr1>
esf<Functor, Grid, boost::mpl::vector<PlcHldr0, PlcHldr1> >
make_esf(PlcHldr0, PlcHldr1) {
    return esf<Functor, Grid, boost::mpl::vector<PlcHldr0, PlcHldr1>>();
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
public:
    using storage_types = typename std::remove_const<typename boost::fusion::result_of::as_vector<typename GridType::storage_types>::type>::type;
    
    using mpl_pointers_t_ = typename boost::mpl::transform<PlcVector,
                                                           get_pointer<GridType>
                                                            >::type;

    using pointers_t = typename boost::fusion::result_of::as_vector<mpl_pointers_t_>::type;

private:
    storage_types storages;
    pointers_t pointers;
    GridType const& grid;
    unsigned int m_pos_i;
    unsigned int m_pos_j;

    template <typename PointersT, typename StoragesT>
    struct _set_pointers {
        PointersT &pt;
        StoragesT const &st;
        _set_pointers(PointersT& pt, StoragesT const &st): pt(pt), st(st) {}

        template <typename Index>
        void operator()(Index) {
            boost::fusion::at_c<Index::value>(pt) = const_cast<double*>((boost::fusion::at_c<Index::value>(st))->min_addr());
        }
    };
    
public:
    accessor_type(storage_types const& storages, GridType const& grid, unsigned int pos_i, unsigned int pos_j)
        : storages(storages)
        , grid(grid)
        , m_pos_i(pos_i)
        , m_pos_j(pos_j)
    {
        using indices = typename boost::mpl::range_c<int, 0, boost::fusion::result_of::size<storage_types>::type::value >;
        boost::mpl::for_each<indices>(_set_pointers<pointers_t, storage_types>(pointers, storages));
    }

    void inc_i() {++m_pos_i;}
    void inc_j() {++m_pos_j;}
    void reset_i() {m_pos_i = 0;}
    void reset_j() {m_pos_j = 0;}
    void set_ij(int i, int j) {
        m_pos_i = i;
        m_pos_j = j;
    }
    unsigned int i() const {return m_pos_i;}
    unsigned int j() const {return m_pos_j;}

    template <typename Arg, typename Accumulator>
    double operator()(on_cells_impl<Arg, Accumulator> oncells, double initial = 0.0) const {
        auto neighbors = grid.cell2cells(*(boost::fusion::at_c<Arg::value>(storages)), {m_pos_i, m_pos_j});
        double result = initial;

        for (int i = 0; i<neighbors.size(); ++i) {
            result = oncells.ff(*(boost::fusion::at_c<Arg::value>(pointers)), result);
        }
    }

    template <int I>
    double& operator()(accessor<I> const& arg) const {
        return *(boost::fusion::at_c<I>(pointers));
    }
};


namespace gridtools {
    template <typename T, size_t D>
    std::ostream& operator<<(std::ostream &s, array<T,D,void> const& a) {
        s << " {  ";
        for (int i = 0; i < D-1; ++i) {
            s << a[i] << ", ";
        }
        s << a[D-1] << "  } ";

        return s;
    }
}

struct on_cells_f {
    typedef accessor<0> out;
    typedef accessor<1> in;
    
    template <typename GridAccessors>
    void
    operator()(GridAccessors /*const*/& eval/*, region*/) const {
        std::cout << "i = " << eval.i()
                  << " j = " << eval.j()
                  << std::endl;
        auto ff = [](const double _in, const double _res) -> double {return _in+_res;};
        eval(out()) = eval(on_cells<in>(ff), 0.0);
    }
};

#define EVAL(f,storage,x,y)                                             \
    std::cout << #f << ": " << gridtools::array<int,2>{x,y} << " -> " << (grid.f(storage,{x,y})) << std::endl

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

    std::cout << trapezoid_2D().u_cell_size(gridtools::array<unsigned int, 2>{NC, MC}) << std::endl;
    std::cout << trapezoid_2D().u_edge_size(gridtools::array<unsigned int, 2>{NC, MC}) << std::endl;
    
    cell_storage_type cells(trapezoid_2D().u_cell_size(gridtools::array<unsigned int, 2>{NC, MC}));
    edge_storage_type edges(trapezoid_2D().u_edge_size(gridtools::array<unsigned int, 2>{NE, ME}));

    trapezoid_2D grid(/*cells, edges,*/ NC, MC);

    cells.info();
    
    EVAL(cell2cells_ll_p0, cells, 1, 1);
    EVAL(cell2cells_ll_p0, cells, 1, 2);
    EVAL(cell2cells_ll_p1, cells, 1, 3);
    EVAL(cell2cells_ll_p1, cells, 1, 4);
    EVAL(cell2cells, cells, 2, 3);
    EVAL(cell2cells, cells, 2, 4);
    EVAL(cell2cells, cells, 3, 3);
    EVAL(cell2cells, cells, 3, 4);

    EVAL(edge2edges_ll_p0, edges, 2, 3);
    EVAL(edge2edges_ll_p1, edges, 2, 3);
    EVAL(edge2edges_ll_p2, edges, 2, 3);
    EVAL(edge2edges, edges, 2, 2);
    EVAL(edge2edges, edges, 2, 3);
    EVAL(edge2edges, edges, 2, 4);

    EVAL(cell2edges_ll_p0, edges, 2, 3);
    EVAL(cell2edges_ll_p1, edges, 2, 3);
    EVAL(cell2edges, edges, 2, 3);
    EVAL(cell2edges, edges, 2, 4);

    EVAL(edge2cells_ll_p0, cells, 2, 3);
    EVAL(edge2cells_ll_p1, cells, 2, 3);
    EVAL(edge2cells_ll_p2, cells, 2, 3);
    EVAL(edge2cells, cells, 2, 3);
    EVAL(edge2cells, cells, 2, 4);
    EVAL(edge2cells, cells, 2, 5);


    cell_storage_type cells_out(trapezoid_2D().u_cell_size(gridtools::array<unsigned int, 2>{NC, MC}));
    edge_storage_type edges_out(trapezoid_2D().u_edge_size(gridtools::array<unsigned int, 2>{NE, ME}));

    //    trapezoid_2D grid_out(cells_out, edges_out, NC, MC);

    typedef arg<0, trapezoid_2D::cells> in_cells;
    typedef arg<1, trapezoid_2D::cells> out_cells;

    auto x = make_esf<on_cells_f, trapezoid_2D>(in_cells(), out_cells());

    accessor_type<boost::mpl::vector<in_cells, out_cells>, trapezoid_2D> acc
        (boost::fusion::vector<cell_storage_type*, cell_storage_type*>(&cells_out, &cells), grid, 0,0);

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
