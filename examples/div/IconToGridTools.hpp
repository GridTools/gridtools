//
// Created by Xiaolin Guo on 13.04.16.
//

#pragma once

#include <netcdf>
#include <stencil-composition/stencil-composition.hpp>
#include <unordered_map>

class IconToGridToolsBase
{
    netCDF::NcFile dataFile_;
    template<typename T>
    std::vector<std::vector<T>> get2DVarTranspose(std::string varName) const;
    void buildMapping();

protected:
    typedef std::unordered_map<int, std::tuple<int, int, int>> i2g_t; // tuple<i, c, j>
    void printi2g(i2g_t&);
    i2g_t i2g_vertex;
    i2g_t i2g_edge;
    i2g_t i2g_cell;

    int length_;

    IconToGridToolsBase(char *ncFileName);

    template<typename T>
    std::vector<T> get1DVar(std::string varName) const;
};

template<typename T>
std::vector<T> IconToGridToolsBase::get1DVar(std::string varName) const
{
    netCDF::NcVar var = dataFile_.getVar(varName);
    size_t dim = var.getDims()[0].getSize();
    std::vector<T> vec(dim);
    var.getVar(vec.data());

    return vec;
}

template<typename IcosahedralTopology>
class IconToGridTools: protected IconToGridToolsBase
{
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

    template<typename LocationType, typename ValueType>
    decltype(auto) get(char const *name);

    IcosahedralTopology &icosahedral_grid() {return icosahedral_grid_;}
};

template<typename IcosahedralTopology>
IconToGridTools<IcosahedralTopology>::IconToGridTools(char *ncFileName)
    : IconToGridToolsBase(ncFileName),
      icosahedral_grid_(length_, length_, 1)
{ }

template<typename IcosahedralTopology>
template<typename LocationType, typename ValueType>
decltype(auto) IconToGridTools<IcosahedralTopology>::get(char const *name)
{
    auto field = icosahedral_grid_.template make_storage<LocationType, ValueType>(name);
    field.initialize(0.0);

    std::vector<ValueType> icon_field = get1DVar<ValueType>(name);

    auto i2s_vector = get_i2g_vector(TypeHelper<LocationType>());

    for (int idx = 0; idx < icon_field.size(); ++idx)
    {
        auto it = i2s_vector.find(idx + 1);
        if (it != i2s_vector.end())
        {
            int i, c, j;
            std::tie(i, c, j) = it->second;
            field(i, c, j, 0) = icon_field[idx];
        }
    }

    return field;
};

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
