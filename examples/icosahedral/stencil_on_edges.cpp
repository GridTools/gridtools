#include "gtest/gtest.h"
#include "stencil_on_edges.hpp"
#include "../Options.hpp"

int main(int argc, char **argv) {

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 4) {
        printf("Usage: copy_stencil_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n");
        return 1;
    }

    for (int i = 0; i != 3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i + 1]);
    }

    if (argc > 4) {
        Options::getInstance().m_size[3] = atoi(argv[4]);
    }
    if (argc == 6) {
        if ((std::string(argv[5]) == "-d"))
            Options::getInstance().m_verify = false;
    }

    return RUN_ALL_TESTS();
}

TEST(StencilOnEdges, Test) {
    uint_t x = Options::getInstance().m_size[0];
    uint_t y = Options::getInstance().m_size[1];
    uint_t z = Options::getInstance().m_size[2];
    uint_t t = Options::getInstance().m_size[3];
    bool verify = Options::getInstance().m_verify;

    if (t == 0)
        t = 1;

    ASSERT_TRUE(soe::test(x, y, z, t, verify));
}
