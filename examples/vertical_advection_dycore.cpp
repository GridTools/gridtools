#include "gtest/gtest.h"
#include "Options.hpp"
#include "vertical_advection_dycore.hpp"

int main(int argc, char** argv)
{
    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 4) {
        printf( "Usage: vertical_advection_dycore_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n" );
        return 1;
    }

    return !vertical_advection_dycore::test(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
}
