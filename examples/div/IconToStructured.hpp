//
// Created by Xiaolin Guo on 13.04.16.
//

#pragma once

#include <netcdf>


class IconToStructured
{
    template<typename T>
    std::vector<T> get1DVar(std::string varName, netCDF::NcFile &dataFile);
    template<typename T>
    std::vector<std::vector<T>> get2DVarTranspose(std::string varName, netCDF::NcFile &dataFile);
    void buildMapping(netCDF::NcFile &dataFile);

    std::vector<int> i2s_vertex;
    std::vector<int> i2s_edge;
    std::vector<int> i2s_cell;
public:
    IconToStructured(char *ncFileName);
};

