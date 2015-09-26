#include "extended_4D.hpp"
#include <iostream>

int main(int argc, char** argv)
{

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 4) {
        printf( "Usage: extended_4D_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n" );
        return 1;
    }

    for(int i=0; i!=3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i+1]);
    }

    return RUN_ALL_TESTS();
}
