#include "Options.hpp"
#include "gtest/gtest.h"
#include "expandable_parameters.hpp"

int main(int argc, char **argv) {

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 4) {
        printf("Usage: expandable_parameters_<whatever> dimx dimy dimz\n where args are integer sizes of the data "
               "fields\n");
        return 1;
    }

    for (int i = 0; i != 3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i + 1]);
    }

    if (argc == 5) {
        Options::getInstance().m_size[3] = atoi(argv[4]);
    }

    return RUN_ALL_TESTS();
}

TEST(ExpandableParameters, Test) {
    gridtools::uint_t x = Options::getInstance().m_size[0];
    gridtools::uint_t y = Options::getInstance().m_size[1];
    gridtools::uint_t z = Options::getInstance().m_size[2];
    gridtools::uint_t t = Options::getInstance().m_size[3];
    if (t == 0)
        t = 1;

    ASSERT_TRUE(test_expandable_parameters::test(x, y, z, t));
}
