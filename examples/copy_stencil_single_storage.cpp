#include "gtest/gtest.h"
#include "copy_stencil_single_storage.hpp"
#include "Options.hpp"

TEST(CopyStencil, SingleStorageTest)
{
    uint_t x = Options::getInstance().m_size[0];
    uint_t y = Options::getInstance().m_size[1];
    uint_t z = Options::getInstance().m_size[2];

    ASSERT_TRUE(copy_stencil::test(x, y, z));
}
