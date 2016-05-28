//
// Created by Xiaolin Guo on 13.04.16.
//

#pragma once

#include <netcdf>
#include <stencil-composition/stencil-composition.hpp>
// nvcc does not recognize std::unordered_map
#include <tr1/unordered_map>

class IconToGridToolsBase
{
    netCDF::NcFile dataFile_;
    void buildMapping();

protected:
    typedef std::tr1::unordered_map<int, std::tuple<int, int, int>> i2g_t; // tuple<i, c, j>
    void printi2g(i2g_t&);
    i2g_t i2g_vertex;
    i2g_t i2g_edge;
    i2g_t i2g_cell;

    unsigned int length_;

    IconToGridToolsBase(char *ncFileName);

    template<typename T>
    std::vector<T> get1DVar(const char *varName) const;
    template<typename T>
    std::vector<std::vector<T>> get2DVarTranspose(const char *varName) const;
};

template<typename T>
std::vector<T> IconToGridToolsBase::get1DVar(const char *varName) const
{
    netCDF::NcVar var = dataFile_.getVar(varName);
    size_t dim = var.getDims()[0].getSize();
    std::vector<T> vec(dim);
    var.getVar(vec.data());

    return vec;
}

template<typename T>
std::vector<std::vector<T>> IconToGridToolsBase::get2DVarTranspose(const char *varName) const
{
    using std::vector;

    netCDF::NcVar var = dataFile_.getVar(varName);
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

template<typename IcosahedralTopology>
class IconToGridTools: protected IconToGridToolsBase
{
    const unsigned int d3_;
    IcosahedralTopology icosahedral_grid_;

    template<typename T>
    struct TypeHelper
    {
    };

    i2g_t &get_i2g_vector(TypeHelper<typename IcosahedralTopology::edges>);
    i2g_t &get_i2g_vector(TypeHelper<typename IcosahedralTopology::cells>);
    i2g_t &get_i2g_vector(TypeHelper<typename IcosahedralTopology::vertexes>);
public:
    IconToGridTools(char *ncFileName);

    // We cannot define this method out-of line
    // This seems to be a GCC bug that it fails to match out-of-line template member function definition with declaration
    template<typename LocationType, typename ValueType>
    auto get(char const *name) -> decltype(std::declval<IcosahedralTopology>().template make_storage<LocationType, ValueType>(name))
    {
        auto field = icosahedral_grid_.template make_storage<LocationType, ValueType>(name);
        field.initialize(0.0);

        std::vector<ValueType> icon_field = get1DVar<ValueType>(name);

        auto& i2g_vector = get_i2g_vector(TypeHelper<LocationType>());

        for (int idx = 0; idx < icon_field.size(); ++idx)
        {
            auto it = i2g_vector.find(idx + 1);
            if (it != i2g_vector.end())
            {
                int i, c, j;
                std::tie(i, c, j) = it->second;
                for (int k = 0; k < d3_; ++k)
                    field(i, c, j, k) = icon_field[idx];
            }
        }

        return field;
    }

    template<typename StorageType, typename LocationType, typename MetaType>
    StorageType get(MetaType& meta, const char *name);

    IcosahedralTopology &icosahedral_grid() {return icosahedral_grid_;}
    unsigned int d3() {return d3_;}
};

template<typename IcosahedralTopology>
IconToGridTools<IcosahedralTopology>::IconToGridTools(char *ncFileName)
    : IconToGridToolsBase(ncFileName),
      d3_(50),
      icosahedral_grid_(length_, length_, d3_)
{ }

template<typename IcosahedralTopology>
typename IconToGridTools<IcosahedralTopology>::i2g_t &IconToGridTools<IcosahedralTopology>::get_i2g_vector(TypeHelper<
    typename IcosahedralTopology::edges>)
{ return i2g_edge; }

template<typename IcosahedralTopology>
typename IconToGridTools<IcosahedralTopology>::i2g_t &IconToGridTools<IcosahedralTopology>::get_i2g_vector(TypeHelper<
    typename IcosahedralTopology::cells>)
{ return i2g_cell; }

template<typename IcosahedralTopology>
typename IconToGridTools<IcosahedralTopology>::i2g_t &IconToGridTools<IcosahedralTopology>::get_i2g_vector(TypeHelper<
        typename IcosahedralTopology::vertexes>)
{ return i2g_vertex; }

// TODO: ICON may store edges of vertexes in a different order than GridTools
template<typename IcosahedralTopology>
template<typename StorageType, typename LocationType, typename MetaType>
StorageType IconToGridTools<IcosahedralTopology>::get(MetaType& meta, const char *name)
{
    StorageType field(meta, name);
    field.initialize(0.0);

    auto icon_field = get2DVarTranspose<double>(name);
//    auto edges_of_vertex = get2DVarTranspose<int>("edges_of_vertex");
    auto& i2g_vector = get_i2g_vector(TypeHelper<LocationType>());

    for (int idx = 0; idx < icon_field.size(); ++idx)
    {
        auto it = i2g_vector.find(idx + 1);
        if (it == i2g_vector.end())
            continue;

        int i, c, j;
        std::tie(i, c, j) = it->second;
        for (int edge = 0; edge < icon_field[idx].size(); ++edge)
        {
            for (int k = 0; k < d3_; ++k)
                field(i, c, j, k, edge) = icon_field[idx][edge];
        }
    }

    return field;
}
