#include "gtest/gtest.h"
#include "Options.hpp"
#include "interface1_functions.hpp"

#ifdef FUNCTIONS_MONOLITHIC
#define FTESTNAME(x) HorizontalDiffusionFunctionsMONOLITHIC
#endif

#ifdef FUNCTIONS_CALL
#define FTESTNAME(x) HorizontalDiffusionFunctionsCALL
#endif

#ifdef FUNCTIONS_OFFSETS
#define FTESTNAME(x) HorizontalDiffusionFunctionsOFFSETS
#endif

#ifdef FUNCTIONS_PROCEDURES
#define FTESTNAME(x) HorizontalDiffusionFunctionsPROCEDURES
#endif

#ifdef FUNCTIONS_PROCEDURES_OFFSETS
#define FTESTNAME(x) HorizontalDiffusionFunctionsPROCEDURESOFFSETS
#endif


int main(int argc, char** argv)
{
    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 4) {
        printf( "Usage: interface1_<whatever> dimx dimy dimz tsteps \n where args are integer sizes of the data fields and tstep is the number of timesteps to run in a benchmark run\n" );
        return 1;
    }

    for(int i=0; i!=3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i+1]);
    }

    if(argc==5) {
        Options::getInstance().m_size[3] = atoi(argv[4]);
    }

    return RUN_ALL_TESTS();
}


TEST(FTESTNAME(x), Test)
{
    uint_t x = Options::getInstance().m_size[0];
    uint_t y = Options::getInstance().m_size[1];
    uint_t z = Options::getInstance().m_size[2];
    uint_t t = Options::getInstance().m_size[3];
    if(t==0) t=1;

    ASSERT_TRUE(horizontal_diffusion_functions::test(x, y, z, t));
}
