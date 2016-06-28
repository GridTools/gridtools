#include "gtest/gtest.h"
#include "Options.hpp"
#include "simple_hori_diff.hpp"

int main(int argc, char** argv)
{

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 4) {
        printf( "Usage: interface1_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n" );
        return 1;
    }

    for(int i=0; i!=3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i+1]);
    }

    if(argc==5) {
        Options::getInstance().m_size[3] = atoi(argv[4]);
    }
    if (argc == 6) {
        if((std::string(argv[5]) == "-d"))
            Options::getInstance().m_verify = false;
    }

    return RUN_ALL_TESTS();
}

TEST(HorizontalDiffusion, Test)
{
    uint_t x = Options::getInstance().m_size[0];
    uint_t y = Options::getInstance().m_size[1];
    uint_t z = Options::getInstance().m_size[2];
    uint_t t = Options::getInstance().m_size[3];
    bool verify = Options::getInstance().m_verify;
    if(t==0) t=1;

    ASSERT_TRUE(shorizontal_diffusion::test(x, y, z, t, verify));
}
