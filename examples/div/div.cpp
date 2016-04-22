#include "gtest/gtest.h"
#include "div.hpp"
#include "../Options.hpp"

int main(int argc, char **argv)
{
    Options::getInstance().mesh_file = argv[1];

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 0; i != 3; ++i) {
        Options::getInstance().m_size[i] = 10;
    }

    return RUN_ALL_TESTS();
}

TEST(DivStencil, Test) {
    uint_t x = Options::getInstance().m_size[0];
    uint_t y = Options::getInstance().m_size[1];
    uint_t z = Options::getInstance().m_size[2];
    uint_t t = Options::getInstance().m_size[3];
    if (t == 0)
        t = 1;

    ASSERT_TRUE(divergence::test(x, y, z, t, Options::getInstance().mesh_file));
}
