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
    std::vector<std::vector<T>> get2DVarTranspose(const char *varName) const;
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
    std::vector<T> get1DVar(const char *varName) const;
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

template<typename IcosahedralTopology>
class IconToGridTools: protected IconToGridToolsBase
{
    const int d3_;
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

        auto& i2s_vector = get_i2g_vector(TypeHelper<LocationType>());

        for (int idx = 0; idx < icon_field.size(); ++idx)
        {
            auto it = i2s_vector.find(idx + 1);
            if (it != i2s_vector.end())
            {
                int i, c, j;
                std::tie(i, c, j) = it->second;
                for (int k = 0; k < d3_; ++k)
                    field(i, c, j, k) = icon_field[idx];
            }
        }

        return field;
    };


    IcosahedralTopology &icosahedral_grid() {return icosahedral_grid_;}
    int d3() {return d3_;}
};

template<typename IcosahedralTopology>
IconToGridTools<IcosahedralTopology>::IconToGridTools(char *ncFileName)
    : IconToGridToolsBase(ncFileName),
      d3_(3),
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
