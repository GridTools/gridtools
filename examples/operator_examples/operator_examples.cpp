#include "div.hpp"
#include "curl.hpp"
#include "../Options.hpp"

int main(int argc, char **argv)
{
    Options::getInstance().mesh_file = argv[1];

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc == 3) {
        Options::getInstance().m_size[3] = atoi(argv[2]);
    }

    return RUN_ALL_TESTS();
}

TEST(DivStencil, Test) {
    gridtools::uint_t t = Options::getInstance().m_size[3];
    if (t == 0)
        t = 1;

    ASSERT_TRUE(operator_examples::test_div(t, Options::getInstance().mesh_file));
}

TEST(CurlStencil, Test) {
    int t = Options::getInstance().m_size[3];
    if (t == 0)
        t = 1;

    ASSERT_TRUE(operator_examples::test_curl(t, Options::getInstance().mesh_file));
}

