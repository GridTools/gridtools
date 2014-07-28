#include <iostream>
#include <storage/storage.h>
#include <common/layout_map.h>
#include <fstream>
#include <boost/timer/timer.hpp>

int main(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    /**
	The following steps are performed:

	- Definition of the domain:
    */
    int d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    int d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    int d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    typedef gridtools::storage<double, gridtools::layout_map<0,1,2> > storage_type;
    std::ofstream file_i("basic_in");
    std::ofstream file_o("basic_out");

    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    out.print(file_i);

    boost::timer::cpu_timer time;
    for (int i=2; i < d1-2; ++i) {
        for (int j=2; j < d2-2; ++j) {
            for (int k=0; k < d3; ++k) {
                //std::cout << in(i,j,k) << std::endl;
                out(i,j,k) = 4 * in(i,j,k) - 
                    (in( i+1, j, k) + in( i, j+1, k) +
                     in( i-1, j, k) + in( i, j-1, k));
            }
        }
    }
    boost::timer::cpu_times lapse_time = time.elapsed();

    out.print(file_o);

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

    return 0;
}
