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
    neighbor_offsets_2D(int a) {
        m_offset[0] = 1;
        m_offset[1] = a;
    }

    int offset(int neighbor_index) const {
        return (neighbor_index&1)?m_offset[neighbor_index>>1]:-m_offset[neighbor_index>>1];
    }
};


struct sphered_cube {
    struct edge {
        int face;
        int edge_index;
        bool forward;
        std::vector<double> buffer;

        edge (int f, int e, bool d, int s)
            : face(f)
            , edge_index(e)
            , forward(d)
            , buffer()
        {}
    };

    int m_sizes[3];

    structured_storage<neighbor_offsets_2D>* faces[6];

    std::vector<std::vector<edge> > face_structure;

    sphered_cube(int n, int m, int l)
    {
        m_sizes[0] = n+2;
        m_sizes[1] = m+2;
        m_sizes[2] = l+2;

        for (int l_face=0; l_face<6; ++l_face) {
            auto ss = sizes(l_face);

            faces[l_face] = new structured_storage<neighbor_offsets_2D>
                (std::vector<double>(ss.first*ss.second), neighbor_offsets_2D(ss.second+2));
        }

        {
            int l_face = 0;
            int selector = l_face&1;
            int size0 = l_face>>1;
            int size1 = (size0+1)%3;
            std::vector<edge> edges0;
            edges0.push_back(edge(5, 1, true, size1));
            edges0.push_back(edge(3, 2, true, size0));
            edges0.push_back(edge(2, 1, true, size0));
            edges0.push_back(edge(4, 1, false, size1));
            face_structure.push_back(edges0);
        }
        {
            int l_face = 1;
            int selector = l_face&1;
            int size0 = l_face>>1;
            int size1 = (size0+1)%3;
            std::vector<edge> edges1;
            edges1.push_back(edge(5, 2, false, size1));
            edges1.push_back(edge(2, 2, true, size0));
            edges1.push_back(edge(3, 1, true, size0));
            edges1.push_back(edge(4, 2, true, size1));
            face_structure.push_back(edges1);
        }
        {
            int l_face = 2;
            int selector = l_face&1;
            int size0 = l_face>>1;
            int size1 = (size0+1)%3;
            std::vector<edge> edges2;
            edges2.push_back(edge(5, 3, true, size1));
            edges2.push_back(edge(0, 2, true, size0));
            edges2.push_back(edge(1, 1, true, size0));
            edges2.push_back(edge(4, 0, true, size1));
            face_structure.push_back(edges2);
        }
        {
            int l_face = 3;
            int selector = l_face&1;
            int size0 = l_face>>1;
            int size1 = (size0+1)%3;
            std::vector<edge> edges3;
            edges3.push_back(edge(5, 0, false, size1));
            edges3.push_back(edge(1, 2, true, size0));
            edges3.push_back(edge(0, 1, true, size0));
            edges3.push_back(edge(4, 3, false, size1));
            face_structure.push_back(edges3);
        }
        {
            int l_face = 4;
            int selector = l_face&1;
            int size0 = l_face>>1;
            int size1 = (size0+1)%3;
            std::vector<edge> edges4;
            edges4.push_back(edge(2, 3, true, size1));
            edges4.push_back(edge(0, 3, false, size0));
            edges4.push_back(edge(2, 3, true, size0));
            edges4.push_back(edge(3, 3, false, size1));
            face_structure.push_back(edges4);
        }
        {
            int l_face = 5;
            int selector = l_face&1;
            int size0 = l_face>>1;
            int size1 = (size0+1)%3;
            std::vector<edge> edges5;
            edges5.push_back(edge(3, 0, false, size1));
            edges5.push_back(edge(0, 0, true, size0));
            edges5.push_back(edge(1, 0, false, size0));
            edges5.push_back(edge(2, 0, true, size1));
            face_structure.push_back(edges5);
        }

    }

    structured_storage<neighbor_offsets_2D>& face(int i) {
        return *faces[i];
    }

    // structured_storage<neighbor_offsets_2D>& neighbor_face(int i, int side_index, int side_parity) {
    //     int offset = (side_index==0)?2:4;
    //     if (side_parity) {
    //         std::cout << "    parity " << side_parity 
    //                   << ", face " << i
    //                   << " : ((i&110)+offset)%6 = "
    //                   << "((" << i << "&110)+offset)%6 = "
    //                   << "(" << (i&110) << "+offset)%6 = "
    //                   << "(" << (i&110)+offset << ")%6 = "
    //                   << ((i&110)+offset)%6
    //                   << std::endl;;
    //         return *faces[((i&110)+offset)%6];
    //     } else {
    //         std::cout << "    parity " << side_parity 
    //                   << ", face " << i
    //                   << " : ((i&110)+offset+1)%6 = "
    //                   << "((" << i << "&110)+offset+1)%6 = "
    //                   << "(" << (i&110) << "+offset+1)%6 = "
    //                   << "(" << (i&110)+offset+1 << ")%6 = "
    //                   << ((i&110)+offset+1)%6
    //                   << std::endl;;
    //         return *faces[((i&110)+offset+1)%6];
    //     }
    // }

    std::pair<int, int> sizes(int i) const {
        int size0, size1;
        switch (i>>1) {
        case 0:
            size0=0;
            size1=1;
            break;
        case 1:
            size0=0;
            size1=2;
            break;
        case 2:
            size0=1;
            size1=2;
            break;
        }

        return std::make_pair(m_sizes[size0], m_sizes[size1]);
    }

    void extract_edges() {
        for (int f = 0; f < 6; ++f) {
            for (int e = 0; e < 4; ++e) {
                auto ss = sizes(face_structure[f][e].face);
                if (face_structure[f][e].forward) {
                    switch (face_structure[f][e].edge_index) {
                    case 0:
                        for (int i=1; i<2; ++i) {
                            for (int j=1; j<ss.second-1; ++j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    case 1:
                        for (int i=1; i<ss.first-1; ++i) {
                            for (int j=1; j<2; ++j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    case 2:
                        for (int i=1; i<ss.first-1; ++i) {
                            for (int j=ss.second-2; j<ss.second-1; ++j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    case 3:
                        for (int i=ss.first-2; i<ss.first-1; ++i) {
                            for (int j=1; j<ss.second-1; ++j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    default :
                        exit(1);
                    }
                } else {
                    switch (face_structure[f][e].edge_index) {
                    case 0:
                        for (int i=1; i<2; ++i) {
                            for (int j=ss.second-2; j>0; --j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    case 1:
                        for (int i=ss.first-2; i>0; --i) {
                            for (int j=1; j<2; ++j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    case 2:
                        for (int i=ss.first-2; i>0; --i) {
                            for (int j=ss.second-2; j<ss.second-1; ++j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    case 3:
                        for (int i=ss.first-2; i<ss.first-1; ++i) {
                            for (int j=ss.second-2; j>0; --j) {
                                face_structure[f][e].buffer.push_back(faces[face_structure[f][e].face]->data[i*ss.second+j]);
                            }
                        }
                        break;
                    default :
                        exit(1);
                    }
                }
            }
        }
    }

    void update_edges() {
        for (int f = 0; f < 6; ++f) {
            auto ss = sizes(f);
            {
                int e=0;
                int k=0;
                for (int i=0; i<1; ++i) {
                    for (int j=1; j<ss.second-1; ++j) {
                        faces[f]->data[i*ss.second+j] = 
                            face_structure[f][e].buffer[k++];
                    }
                }
            }

            {
                int e=1;
                int k=0;
                for (int i=1; i<ss.first-1; ++i) {
                    for (int j=0; j<1; ++j) {
                        faces[f]->data[i*ss.second+j] = 
                            face_structure[f][e].buffer[k++];
                    }
                }
            }

            {
                int e=2;
                int k=0;
                for (int i=1; i<ss.first-1; ++i) {
                    for (int j=ss.second-1; j<ss.second; ++j) {
                        faces[f]->data[i*ss.second+j] = 
                            face_structure[f][e].buffer[k++];
                    }
                }
            }

            {
                int e=3;
                int k=0;
                for (int i=ss.first-1; i<ss.first; ++i) {
                    for (int j=1; j<ss.second-1; ++j) {
                        faces[f]->data[i*ss.second+j] = 
                            face_structure[f][e].buffer[k++];
                    }
                }
            }
        }
    }



    void print(int halo=0) {
        int n = m_sizes[0];
        int m = m_sizes[1];
        int l = m_sizes[2];

        std::cout << "n = " << n-2+2*halo << ", ";
        std::cout << "m = " << m-2+2*halo << ", ";
        std::cout << "l = " << l-2+2*halo << ".\n\n";
        std::cout << "           --------\n";
        std::cout << "           |      |\n";
        std::cout << "           |  4   |\n";
        std::cout << "       m   |  l   |\n";
        std::cout << "    -----------------------------\n";
        std::cout << "    |      |      |      |      |\n";
        std::cout << "  n |  0   |  2   |  1   |  3   |\n";
        std::cout << "    |      |      |      |      |\n";
        std::cout << "    -----------------------------\n";
        std::cout << "           |      |\n";
        std::cout << "           |  5   |\n";
        std::cout << "           |      |\n";
        std::cout << "           --------\n";

        for (int l_face = 0; l_face < 6; ++l_face) {
            std::cout << "FACE " << l_face << std::endl;
            auto l_s = sizes(l_face);
            std::cout << "sizes "
                  << l_s.first << " " 
                  << l_s.second << " " 
                  << std::endl;

            for (int i=1-halo; i<l_s.first-1+halo; i+=std::max(1,l_s.first/12)) {
                for (int j=1-halo; j<l_s.second-1+halo; j+=std::max(1,l_s.second/12)) {
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
                face.data[i*sizes.second+j]=(f+1)*(i*sizes.second+j);
                lap_face.data[i*sizes.second+j]=(f+1)*(i*sizes.second+j);
            }
        }
    }

    storage.print();

    storage.extract_edges();

    for (int f=0; f<6; ++f) {
        std::cout << "Face: " << f;
        for (int s=0; s<4; ++s) {
            std::cout << "     Neighbor: " << s << "\n"
                      << "     Face ID : " << storage.face_structure[f][s].face << "\n"
                      << "     EdgeID  : " << storage.face_structure[f][s].edge_index << "\n"
                      << "     Forward : " << std::boolalpha << storage.face_structure[f][s].forward << "\n"
                      << "     Buffer  : ";
            for (int i=0; i<storage.face_structure[f][s].buffer.size(); ++i) {
                std::cout << storage.face_structure[f][s].buffer[i] << ", ";
            }
            std::cout << std::endl;
        }
    }

    storage.print(1);
    storage.update_edges();
    storage.print(1);

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

    lap.print();

    return 0;
}
