#include <vector>
#include <list>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <stdlib.h>
#include <boost/timer/timer.hpp>
#include "triangular_2D.hpp"
#include "NeighbourLists.hpp"

double c_laplace(triangular_storage<triangular_offsets>& storage, std::vector<std::list<int> >& neighbours, const int cellId)
{
    double lap = 3*storage.data[cellId];
    for(std::list<int>::const_iterator it=neighbours[cellId].begin(); it != neighbours[cellId].end(); ++it) {
        lap -= storage.data[*it];
    }
    return lap;
}

int main(int argc, char** argv) {

    const int m=6;
    const int n=6;
    const int triang_offset=m*2+3;
    const int jstride=triang_offset+1;
    const int row_padd=3;
    const int haloSize = 1;

    int size=(n*2+haloSize*2)*(m*2+haloSize*2);

    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "                          CREATING NEIGHBOUR LIST" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;

    std::vector< std::list<int> > neighbours;

    build_neighbour_list<haloSize,n,m>(neighbours);

    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "                          LAPLACIAN WITH TRIANGULAR MESH" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;

    {
        /** Creating the storages */
        triangular_storage<triangular_offsets> storage(std::vector<double>(size), n, m, haloSize, triangular_offsets(triang_offset));
        triangular_storage<triangular_offsets> lap(std::vector<double>(size), n, m, haloSize, triangular_offsets(triang_offset));
        triangular_storage<triangular_offsets> lap_cool(std::vector<double>(size), n, m, haloSize, triangular_offsets(triang_offset));

        for(int i=0; i < size; ++i){
            storage.data[i] = -1;
            lap.data[i]  = -1;
            lap_cool.data[i]  = -1;
        }

        int domain_start = jstride+1;

        for (int i=-1; i<n+1; ++i) {
            for (int j=-1; j<m*2+1; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                storage.data[cellId] = i*2+i*i*j+j/2.0;
            }
        }

        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            as if it was a C program
        */
        for (int i=0; i<n; ++i) {
            for (int j=0; j<m*2; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                lap.data[cellId] = 3*storage.data[cellId];
                for(std::list<int>::const_iterator it=neighbours[cellId].begin(); it != neighbours[cellId].end(); ++it) {
                    lap.data[cellId] -= storage.data[*it];
                }
            }
        }

        for (int i=0; i<n; ++i) {
            for (int j=0; j<m*2; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                lap_cool.data[cellId] = 3*storage.data[cellId];
                lap_cool.data[cellId] -=
                    storage.fold_neighbors(storage.begin()+cellId, [](double state, double value) { return state+value;});
            }
        }

        bool validated=true;
        for (int i=0; i<n; ++i) {
            for (int j=0; j<m*2; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                if ( ! std::fabs((lap_cool.data[cellId] - lap.data[cellId])/lap_cool.data[cellId]) < 1e-10) {
                    validated=false;
                    std::cout << "FAILED!" << cellId << " " << i << " " << j  << " " << lap_cool.data[cellId] << " " << lap.data[cellId] << std::endl;
                }
            }
        }
        if(validated)
            std::cout << "OK!" << std::endl;
    }

    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "                          4th ORDER DIFFUSION WITH TRIANGULAR MESH" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;

    {
        /** Creating the storages */
        triangular_storage<triangular_offsets> storage(std::vector<double>(size), n, m, haloSize,triangular_offsets(triang_offset));
        triangular_storage<triangular_offsets> lap(std::vector<double>(size), n, m, haloSize, triangular_offsets(triang_offset));
        triangular_storage<triangular_offsets> lap_cool(std::vector<double>(size), n, m, haloSize, triangular_offsets(triang_offset));

        for(int i=0; i < size; ++i){
            storage.data[i] = -1;
            lap.data[i]  = -1;
            lap_cool.data[i]  = -1;
        }

        int domain_start = jstride+1;

        //initialize data
        for (int i=-haloSize; i<n+haloSize; ++i) {
            for (int j=-haloSize; j<m*2+haloSize; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                storage.data[cellId] = i+i*i*j+j/2.0;
            }
        }

        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            as if it was a C program
        */
        for (int i=0; i<n; ++i) {
            for (int j=0; j<m*2; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                lap.data[cellId] = 3*storage.data[cellId];
                for(std::list<int>::const_iterator it=neighbours[cellId].begin(); it != neighbours[cellId].end(); ++it) {
                    lap.data[cellId] -= c_laplace(storage, neighbours, *it);
                }
            }
        }

        for (int i=0; i<n; ++i) {
            for (int j=0; j<m*2; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                lap_cool.data[cellId] =
                    storage.fold_2nd_neighbors(storage.begin()+cellId, 3*storage.data[cellId],
                        [](double state, double value) {return state-value;});
            }
        }

        // Apply by hand loop with indirect addressing for exceptional cells
        std::vector<int> exceptionalCells = {17, 98, 28};
        for(std::vector<int>::const_iterator it = exceptionalCells.begin(); it != exceptionalCells.end(); ++it) {
            lap_cool.data[*it] = 3*storage.data[*it];
            for(std::list<int>::const_iterator nit=neighbours[*it].begin(); nit != neighbours[*it].end(); ++nit) {
                lap_cool.data[*it] -= c_laplace(storage, neighbours, *nit);
            }
        }

        bool validated=true;
        for (int i=0; i<n; ++i) {
            for (int j=0; j<m*2; ++j) {
                int cellId = domain_start +i*(jstride)+j;
                if ( ! std::fabs((lap_cool.data[cellId] - lap.data[cellId])/lap_cool.data[cellId]) < 1e-10) {
                    validated=false;
                    std::cout << "FAILED!" << cellId << " " << i << " " << j  << " " << lap_cool.data[cellId] << " " << lap.data[cellId] << std::endl;
                }
            }
        }
        if(validated)
            std::cout << "OK!" << std::endl;
    }

    return 0;
}
