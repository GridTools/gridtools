#include <iostream>
#include "grid.hpp"
#include <storage/base_storage.h>
#include <common/layout_map.h>

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

#define EVAL(f,x,y)                                                     \
    std::cout << #f << ": " << gridtools::array<int,2>{x,y} << " -> " << (grid.f({x,y})) << std::endl

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

    trapezoid_2D grid(cells, edges, NC, MC);

    cells.info();
    
    EVAL(cell2cells_ll_p0, 1, 1);
    EVAL(cell2cells_ll_p0, 1, 2);
    EVAL(cell2cells_ll_p1, 1, 3);
    EVAL(cell2cells_ll_p1, 1, 4);
    EVAL(cell2cells, 2, 3);
    EVAL(cell2cells, 2, 4);
    EVAL(cell2cells, 3, 3);
    EVAL(cell2cells, 3, 4);

    EVAL(edge2edges_ll_p0, 2, 3);
    EVAL(edge2edges_ll_p1, 2, 3);
    EVAL(edge2edges_ll_p2, 2, 3);
    EVAL(edge2edges, 2, 2);
    EVAL(edge2edges, 2, 3);
    EVAL(edge2edges, 2, 4);

    EVAL(cell2edges_ll_p0, 2, 3);
    EVAL(cell2edges_ll_p1, 2, 3);
    EVAL(cell2edges, 2, 3);
    EVAL(cell2edges, 2, 4);

    // EVAL(cell2edges_offsets, 2, 3);
    // EVAL(cell2edges_offsets, 2, 4);
    // EVAL(cell2edges_offsets, 3, 3);
    // EVAL(cell2edges_offsets, 3, 4);

    return 0;
}
