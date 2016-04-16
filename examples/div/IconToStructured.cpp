//
// Created by Xiaolin Guo on 13.04.16.
//

#include "IconToStructured.hpp"
#include <cassert>
#include <algorithm>

using namespace std;
using namespace netCDF;

IconToStructured::IconToStructured(char *ncFileName)
{
    NcFile dataFile(ncFileName, NcFile::read);

    // description of the netcdf file
    cout << "there are " << dataFile.getVarCount() << " variables" << endl;
    auto vars = dataFile.getVars();
    for (auto &p : vars)
        cout << "  " << p.first << endl;
    cout << "there are " << dataFile.getAttCount() << " attributes" << endl;

    auto atts = dataFile.getAtts();
    for (auto &p : atts)
        cout << "  " << p.first << endl;

    cout << "there are " << dataFile.getDimCount() << " dimensions" << endl;
    auto dims = dataFile.getDims();
    for (auto &p : dims)
        cout << "  " << p.first << endl;

    buildMapping(dataFile);

}
template<typename T>
std::vector<T> IconToStructured::get1DVar(std::string varName, netCDF::NcFile &dataFile)
{
    NcVar var = dataFile.getVar(varName);
    size_t dim = var.getDims()[0].getSize();
    vector<T> vec(dim);
    var.getVar(vec.data());

    return vec;
}
template<typename T>
std::vector<std::vector<T>> IconToStructured::get2DVarTranspose(std::string varName, netCDF::NcFile &dataFile)
{
    NcVar var = dataFile.getVar(varName);
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

void print_i2s_vector(vector<int> i2s_vector)
{
    vector<pair<int, int>> vec;
    for (int i = 0; i < i2s_vector.size(); ++i)
        if (i2s_vector[i] != -1)
            vec.push_back({i2s_vector[i], i});

    sort(vec.begin(), vec.end());
    for (auto &p:vec)
        cout << p.second << " -> " << p.first << endl;
}

void IconToStructured::buildMapping(netCDF::NcFile &dataFile)
{
    vector<vector<int>> vertices_of_vertex = get2DVarTranspose<int>("vertices_of_vertex", dataFile);
    vector<vector<int>> vertex_of_cell = get2DVarTranspose<int>("vertex_of_cell", dataFile);
    vector<vector<int>> edge_of_cell = get2DVarTranspose<int>("edge_of_cell", dataFile);
    vector<vector<int>> edge_vertices = get2DVarTranspose<int>("edge_vertices", dataFile);
    vector<vector<int>> adjacent_cell_of_edge = get2DVarTranspose<int>("adjacent_cell_of_edge", dataFile);

    // we start from topleft corner of the trapezoid
    int left_vertex = 1; // in icon's sense
    int left_cell = 1;
    assert(left_vertex == vertex_of_cell[left_cell - 1][0]);

    i2s_vertex.resize(vertices_of_vertex.size() + 1, -1);
    i2s_cell.resize(vertex_of_cell.size() + 1, -1);
    i2s_edge.resize(edge_vertices.size() + 1, -1);

    auto getNeighborCell = [&adjacent_cell_of_edge, &edge_of_cell](int cell, int edge)
    {
        auto &adjacent_cell = adjacent_cell_of_edge[edge_of_cell[cell - 1][edge] - 1];
        return adjacent_cell[0] == cell ? adjacent_cell[1] : adjacent_cell[0];
    };

    // find edge length of the trapezoid and map the vertices on the top border
    int length = 0;
    {
        int current_vertex = left_vertex;
        int current_cell = left_cell;
        while (length == 0
            || find(vertices_of_vertex[current_vertex - 1].begin(), vertices_of_vertex[current_vertex - 1].end(), 0)
                == vertices_of_vertex[current_vertex - 1].end())
        {
            current_cell = getNeighborCell(current_cell, 1);
            current_cell = getNeighborCell(current_cell, 0);

            current_vertex = vertex_of_cell[current_cell - 1][0];
            ++length;
        }
    }


    // fill i2s_*
    for (int row = 0; row < length; ++row)
    {
        int current_cell = left_cell;

        for (int col{0}; col < length; ++col)
        {
            int current_vertex = vertex_of_cell[current_cell - 1][0];
            i2s_vertex[current_vertex] = row * (length + 1) + col;

            // vertex on bottom line
            if (row == length - 1)
                i2s_vertex[vertex_of_cell[current_cell - 1][1]] = length * (length + 1) + col;

            i2s_edge[edge_of_cell[current_cell - 1][0]] = 3 * row * length + col; // red
            i2s_edge[edge_of_cell[current_cell - 1][1]] = (3 * row + 2) * length + col; // green
            i2s_edge[edge_of_cell[current_cell - 1][2]] = (3 * row + 1) * length + col; // blue
            i2s_cell[current_cell] = row * 2 * length + col;

            current_cell = getNeighborCell(current_cell, 1);
            i2s_cell[current_cell] = (row * 2 + 1) * length + col;

            // last vertex in a line
            if (col == length - 1)
            {
                i2s_vertex[vertex_of_cell[current_cell - 1][1]] = (row + 1) * (length + 1) - 1;
                if (row == length - 1)
                    i2s_vertex[vertex_of_cell[current_cell - 1][0]] = (length + 1) * (length + 1) - 1;
            }

            current_cell = getNeighborCell(current_cell, 0);
        }

        if (row == length - 1)
        {

            int botomleft_vertex = vertex_of_cell[left_cell - 1][1];
            assert(find(vertices_of_vertex[botomleft_vertex - 1].begin(),
                        vertices_of_vertex[botomleft_vertex - 1].end(),
                        0)
                       != vertices_of_vertex[botomleft_vertex - 1].end());
        }

        // move to next line
        left_cell = getNeighborCell(left_cell, 1);
        left_cell = getNeighborCell(left_cell, 2);
    }

    cout << "vertex mapping: " << endl;
    print_i2s_vector(i2s_vertex);

    cout << "edge mapping: " << endl;
    print_i2s_vector(i2s_edge);

    cout << "cell mapping: " << endl;
    print_i2s_vector(i2s_cell);

}










