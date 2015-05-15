#include <iostream>
#include "grid.hpp"

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

int main() {
    using trapezoid_2D = gridtools::trapezoid_2D<>;
    using cell_storage_type = gridtools::cell_storage_type<double>;
    using vertex_storage_type = gridtools::vertex_storage_type<double>;
    using edge_storage_type = gridtools::edge_storage_type<double>;

    trapezoid_2D grid
        (cell_storage_type(trapezoid_2D::u_cell_size_i(20),trapezoid_2D::u_cell_size_j(8)),
         vertex_storage_type(trapezoid_2D::u_vertex_size_i(20),trapezoid_2D::u_vertex_size_j(8)),
         edge_storage_type(trapezoid_2D::u_edge_size_i(20),trapezoid_2D::u_edge_size_j(8)));

    EVAL(cell2cells, 2, 3);
    EVAL(cell2cells, 2, 4);
    EVAL(cell2cells, 3, 3);
    EVAL(cell2cells, 3, 4);

    EVAL(cell2cells_offsets, 2, 3);
    EVAL(cell2cells_offsets, 2, 4);
    EVAL(cell2cells_offsets, 3, 3);
    EVAL(cell2cells_offsets, 3, 4);

    EVAL(vertex2vertices, 2, 3);
    EVAL(vertex2vertices, 2, 4);
    EVAL(vertex2vertices, 3, 3);
    EVAL(vertex2vertices, 3, 4);

    EVAL(vertex2vertices_offsets, 2, 3);
    EVAL(vertex2vertices_offsets, 2, 4);
    EVAL(vertex2vertices_offsets, 3, 3);
    EVAL(vertex2vertices_offsets, 3, 4);

    EVAL(edge2edges, 2, 3);
    EVAL(edge2edges, 2, 4);
    EVAL(edge2edges, 3, 3);
    EVAL(edge2edges, 3, 4);

    EVAL(edge2edges_offsets, 2, 3);
    EVAL(edge2edges_offsets, 2, 4);
    EVAL(edge2edges_offsets, 3, 3);
    EVAL(edge2edges_offsets, 3, 4);

    EVAL(cell2vertices, 2, 3);
    EVAL(cell2vertices, 2, 4);
    EVAL(cell2vertices, 3, 3);
    EVAL(cell2vertices, 3, 4);

    EVAL(cell2vertices_offsets, 2, 3);
    EVAL(cell2vertices_offsets, 2, 4);
    EVAL(cell2vertices_offsets, 3, 3);
    EVAL(cell2vertices_offsets, 3, 4);

    EVAL(cell2edges, 2, 3);
    EVAL(cell2edges, 2, 4);
    EVAL(cell2edges, 3, 3);
    EVAL(cell2edges, 3, 4);

    EVAL(cell2edges_offsets, 2, 3);
    EVAL(cell2edges_offsets, 2, 4);
    EVAL(cell2edges_offsets, 3, 3);
    EVAL(cell2edges_offsets, 3, 4);

    return 0;
}
