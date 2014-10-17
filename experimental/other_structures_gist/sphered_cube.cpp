#include <iostream>
#define INCLUDE_ONLY
#include "cartesian_3D.cpp"

/** This is the function which defines the structure
    i.e., define the offsets of a node.
    In this case the function does not need any
    parameter since the offsets do not depend on
    the current node. The offsets depend on the
    dimensions of the grid.
*/
struct neighbor_offsets_2D {
    int m_offset[2];
    static const int n_neighbors = 4;
    neighbor_offsets_2D(int a, int b) {
        m_offset[0] = 1;
        m_offset[1] = a;
    }

    neighbor_offsets_2D() {
        m_offset[0] = 0;
        m_offset[1] = 0;
    }

    int offset(int neighbor_index) const {
        return (neighbor_index&1)?m_offset[neighbor_index>>1]:-m_offset[neighbor_index>>1];
    }
};


struct sphered_cube {
    int m_sizes[3];
    structured_storage<neighbor_offsets_2D>* faces[6];

    sphered_cube(int n, int m, int l)
    {
        m_sizes[0] = n;
        m_sizes[1] = m;
        m_sizes[2] = l;

        for (int l_face=0; l_face<6; ++l_face) {
            int selector = l_face&1;
            int size0 = l_face>>1;
            int size1 = (size0+1)%3;
            faces[l_face] = new structured_storage<neighbor_offsets_2D>
                (std::vector<double>(m_sizes[size0]*m_sizes[size1]), neighbor_offsets_2D(m*l, l));
        }
    }

    structured_storage<neighbor_offsets_2D>& face(int i) {
        return *faces[i];
    }

    structured_storage<neighbor_offsets_2D>& neighbor_face(int i, int side_index, int side_parity) {
        int offset = (side_index==0)?2:4;
        if (side_parity) {
            std::cout << "    parity " << side_parity 
                      << ", face " << i
                      << " : ((i&110)+offset)%6 = "
                      << "((" << i << "&110)+offset)%6 = "
                      << "(" << (i&110) << "+offset)%6 = "
                      << "(" << (i&110)+offset << ")%6 = "
                      << ((i&110)+offset)%6
                      << std::endl;;
            return *faces[((i&110)+offset)%6];
        } else {
            std::cout << "    parity " << side_parity 
                      << ", face " << i
                      << " : ((i&110)+offset+1)%6 = "
                      << "((" << i << "&110)+offset+1)%6 = "
                      << "(" << (i&110) << "+offset+1)%6 = "
                      << "(" << (i&110)+offset+1 << ")%6 = "
                      << ((i&110)+offset+1)%6
                      << std::endl;;
            return *faces[((i&110)+offset+1)%6];
        }
    }

    std::pair<int, int> sizes(int i) const {
        int size0 = i>>1;
        int size1 = (size0+1)%3;

        return std::make_pair(m_sizes[size0], m_sizes[size1]);
    }


    template <typename Functor>
    double fold_neighbors(iterator it, Functor && f) const {
        double v = 0;

        for (int i=0; i<OffsetFunction::n_neighbors; ++i) {
            v = f(v, it[i]);
        }
        return v;
    }


    void print() {
        int n = m_sizes[0];
        int m = m_sizes[1];
        int l = m_sizes[2];

        std::cout << "n = " << n << ", ";
        std::cout << "m = " << m << ", ";
        std::cout << "l = " << l << ".\n\n";
        std::cout << "           --------\n";
        std::cout << "           |      |\n";
        std::cout << "           |  2   |\n";
        std::cout << "       m   |  l   |\n";
        std::cout << "    -----------------------------\n";
        std::cout << "    |      |      |      |      |\n";
        std::cout << "  n |  0   |  4   |  1   |  5   |\n";
        std::cout << "    |      |      |      |      |\n";
        std::cout << "    -----------------------------\n";
        std::cout << "           |      |\n";
        std::cout << "           |  3   |\n";
        std::cout << "           |      |\n";
        std::cout << "           --------\n";

        for (int l_face = 0; l_face < 6; ++l_face) {
            std::cout << "FACE " << l_face << std::endl;
            auto l_s = sizes(l_face);
            std::cout << "sizes "
                  << l_s.first << " " 
                  << l_s.second << " " 
                  << std::endl;

            for (int i=0; i<l_s.first; i+=std::max(1,l_s.first/5)) {
                for (int j=0; j<l_s.second; j+=std::max(1,l_s.second/5)) {
                    std::cout << std::setw(5) << faces[l_face]->data[i*l_s.second+j] << " ";
                }
                std::cout << std::endl;
            }
        }

    }
};


int main(int argc, char** argv) {
    if (argc==1) {
        std::cout << "Usage: " << argv[0] << " n m l " << std::endl;
        std::cout << "Where n, m, l are the dimensions of the grid" << std::endl;
        return 0;
    }
    int n=atoi(argv[1]);
    int m=atoi(argv[2]);
    int l=atoi(argv[3]);

    sphered_cube storage(n,m,l);
    sphered_cube lap(n,m,l);

    for (int f = 0; f < 6; ++f) {
        auto& face = storage.face(f);
        auto& lap_face = lap.face(f);
        auto sizes = storage.sizes(f);
        std::cout << "sizes "
                  << sizes.first << " " 
                  << sizes.second << " " 
                  << std::endl;
        for(int i=0; i<sizes.first; ++i) {
            for(int j=0; j<sizes.second; ++j) {
                // std::cout << face.data.size() << " < "
                //           << i*sizes.second+j
                //           << std::endl;
                face.data[i*sizes.second+j]=f+1;
                lap_face.data[i*sizes.second+j]=f+1;
            }
        }
    }

    storage.print();

    for (int f = 0; f < 6; ++f) {
        auto& face = storage.face(f);
        auto& lap_face = lap.face(f);
        auto sizes = storage.sizes(f);
        std::cout << "sizes "
                  << sizes.first << " " 
                  << sizes.second << " " 
                  << std::endl;
        for(int i=1; i<sizes.first-1; ++i) {
            for(int j=1; j<sizes.second-1; ++j) {
                lap_face.data[i*sizes.second+j] = 4*face.data[i*sizes.second+j] -
                    face.fold_neighbors(face.begin()+i*sizes.second+j,
                                           [](double state, double value) {
                                               return state+value;
                                           });
            }
        }
    }


    std::cout << "\nBOUNDARIES\n" << std::endl;
    for (int f = 0; f < 6; ++f) {
        auto& face = storage.face(f);
        auto& lap_face = lap.face(f);
        auto sizes = storage.sizes(f);

        std::cout << "FACE: " << f << std::endl;
        std::cout << "sizes "
                  << sizes.first << " " 
                  << sizes.second << " " 
                  << std::endl;

        // Loop over sides
        for (int side_index = 0; side_index < 2; ++side_index) {
              for (int side_parity = 0; side_parity < 2; ++side_parity) {

                  auto& side_face = storage.neighbor_face(f, side_index, side_parity);
                  //auto& lap_side_face = lap.neighbor_face(f, side_index, side_parity);

                  auto side_sizes = storage.sizes(f);

                  std::cout << "side_sizes "
                            << side_sizes.first << " " 
                            << side_sizes.second << " " 
                            << std::endl;

                  face.fold_edge(side_index, side_parity
                
            }
        }
    }

    std::cout << "\nBOUNDARIES DONE\n" << std::endl;

    lap.print();

    return 0;
}
