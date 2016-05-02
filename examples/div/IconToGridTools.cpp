//
// Created by Xiaolin Guo on 13.04.16.
//

#include "IconToGridTools.hpp"
#include <cassert>

using namespace std;
using namespace netCDF;

IconToGridToolsBase::IconToGridToolsBase(char *ncFileName)
    : dataFile_(ncFileName, NcFile::read)
{

//    // description of the netcdf file
//    cout << "there are " << dataFile_.getVarCount() << " variables" << endl;
//    auto vars = dataFile_.getVars();
//    for (auto &p : vars)
//        cout << "  " << p.first << endl;
//    cout << "there are " << dataFile_.getAttCount() << " attributes" << endl;
//
//    auto atts = dataFile_.getAtts();
//    for (auto &p : atts)
//        cout << "  " << p.first << endl;
//
//    cout << "there are " << dataFile_.getDimCount() << " dimensions" << endl;
//    auto dims = dataFile_.getDims();
//    for (auto &p : dims)
//        cout << "  " << p.first << endl;

    int grid_root{0}, grid_level{0};
    dataFile_.getAtt("grid_root").getValues(&grid_root);
    dataFile_.getAtt("grid_level").getValues(&grid_level);

    length_ = grid_root * (1 << grid_level); // grid_root * pow(2, grid_level)

    buildMapping();

}

template<typename T>
std::vector<std::vector<T>> IconToGridToolsBase::get2DVarTranspose(std::string varName) const
{
    NcVar var = dataFile_.getVar(varName);
    size_t dim0 = var.getDims()[0].getSize();
    size_t dim1 = var.getDims()[1].getSize();
    vector<vector<T>> vec(dim0, vector<T>(dim1));
    for (size_t i = 0; i < vec.size(); ++i)
        var.getVar({i, 0}, {1, dim1}, vec[i].data());

    vector<vector<T>> vec_transpose(dim1, vector<T>(dim0));
    for (size_t i = 0; i < vec.size(); ++i)
        for (size_t j = 0; j < vec[0].size(); ++j)
            vec_transpose[j][i] = vec[i][j];

    return vec_transpose;
}

void IconToGridToolsBase::printi2g(i2g_t& vec)
{
    for (const auto& p:vec) {
        int i, c, j;
        std::tie(i, c, j) = p.second;
        cout << p.first << " -> " << "(" << i << ", " << c << ", " << j << ")" << endl;
    }
}

void IconToGridToolsBase::buildMapping()
{
    vector<vector<int>> vertices_of_vertex = get2DVarTranspose<int>("vertices_of_vertex");
    vector<vector<int>> vertex_of_cell = get2DVarTranspose<int>("vertex_of_cell");
    vector<vector<int>> edge_of_cell = get2DVarTranspose<int>("edge_of_cell");
    vector<vector<int>> adjacent_cell_of_edge = get2DVarTranspose<int>("adjacent_cell_of_edge");

    // we start from topleft corner of the trapezoid
    int left_vertex = 1; // in icon's sense
    int left_cell = 1;
    assert(left_vertex == vertex_of_cell[left_cell - 1][0]);

    auto getNeighborCell = [&adjacent_cell_of_edge, &edge_of_cell](int cell, int edge)
    {
        auto &adjacent_cell = adjacent_cell_of_edge[edge_of_cell[cell - 1][edge] - 1];
        return adjacent_cell[0] == cell ? adjacent_cell[1] : adjacent_cell[0];
    };

    // fill i2s_*
    for (int row = 0; row < length_; ++row)
    {
        int current_cell = left_cell;

        for (int col{0}; col < length_; ++col)
        {
            int current_vertex = vertex_of_cell[current_cell - 1][0];
            i2g_vertex[current_vertex] = make_tuple(row, 0, col);

            i2g_edge[edge_of_cell[current_cell - 1][0]] = make_tuple(row, 0, col); // red
            i2g_edge[edge_of_cell[current_cell - 1][2]] = make_tuple(row, 1, col); // blue
            i2g_edge[edge_of_cell[current_cell - 1][1]] = make_tuple(row, 2, col); // green
            i2g_cell[current_cell] = make_tuple(row, 0, col); // downward cell

            current_cell = getNeighborCell(current_cell, 1);
            i2g_cell[current_cell] = make_tuple(row, 1, col); // upward cell

            // TODO: do we really have index for last vertex in a line in gridtools' icosahedral grid?
            // last vertex in a line
            if (col == length_ - 1)
            {
                i2g_vertex[vertex_of_cell[current_cell - 1][1]] = make_tuple(row, 0, length_);
            }

            current_cell = getNeighborCell(current_cell, 0);
        }

        if (row == length_ - 1)
        {

            int botomleft_vertex = vertex_of_cell[left_cell - 1][1];
            assert(find(vertices_of_vertex[botomleft_vertex - 1].begin(),
                        vertices_of_vertex[botomleft_vertex - 1].end(),
                        0)
                       != vertices_of_vertex[botomleft_vertex - 1].end());
        }

        // march to next line
        left_cell = getNeighborCell(left_cell, 1);
        left_cell = getNeighborCell(left_cell, 2);
    }

}
