#include "gtest/gtest.h"

#include "../grid.hpp"
#include "../array_addons.hpp"

using namespace gridtools;

using trapezoid_2D = gridtools::trapezoid_2D_colored<gridtools::_backend>;

class ll_map_test : public ::testing::TestWithParam<int> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
};

TEST_P(ll_map_test, cell_to_cell) {

    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // cell # 50
    ASSERT_TRUE(( grid.ll_map( cells(), cells(), gridtools::static_int<0>(), {3,2}) == array<uint_t, 3>{42*d3, 58*d3, 57*d3} ));
    // cell # 58
    ASSERT_TRUE(( grid.ll_map( cells(), cells(), gridtools::static_int<1>(), {3,2}) == array<uint_t, 3>{50*d3, 51*d3, 66*d3} ));
    // cell # 35
    ASSERT_TRUE(( grid.ll_map( cells(), cells(), gridtools::static_int<0>(), {2,3}) == array<uint_t, 3>{42*d3, 43*d3, 27*d3} ));
    // cell # 45
    ASSERT_TRUE(( grid.ll_map( cells(), cells(), gridtools::static_int<1>(), {2,5}) == array<uint_t, 3>{53*d3, 38*d3, 37*d3} ));

}

TEST_P(ll_map_test, cell_to_edge) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // cell # 50
    ASSERT_TRUE(( grid.ll_map( cells(), edges(), gridtools::static_int<0>(), {3,2}) == array<uint_t, 3>{74*d3, 82*d3, 90*d3} ));
    // cell # 58
    ASSERT_TRUE(( grid.ll_map( cells(), edges(), gridtools::static_int<1>(), {3,2}) == array<uint_t, 3>{90*d3, 75*d3, 106*d3} ));
    // cell # 35
    ASSERT_TRUE(( grid.ll_map( cells(), edges(), gridtools::static_int<0>(), {2,3}) == array<uint_t, 3>{51*d3, 59*d3, 67*d3} ));
    // cell # 45
    ASSERT_TRUE(( grid.ll_map( cells(), edges(), gridtools::static_int<1>(), {2,5}) == array<uint_t, 3>{69*d3, 54*d3, 85*d3} ));

}

TEST_P(ll_map_test, cell_to_vertex) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // cell # 50
    ASSERT_TRUE(( grid.ll_map( cells(), vertexes(), gridtools::static_int<0>(), {3,2}) == array<uint_t, 3>{29*d3, 30*d3, 38*d3} ));
    // cell # 58
    ASSERT_TRUE(( grid.ll_map( cells(), vertexes(), gridtools::static_int<1>(), {3,2}) == array<uint_t, 3>{38*d3, 30*d3, 39*d3} ));
    // cell # 35
    ASSERT_TRUE(( grid.ll_map( cells(), vertexes(), gridtools::static_int<0>(), {2,3}) == array<uint_t, 3>{21*d3, 22*d3, 30*d3} ));
    // cell # 45
    ASSERT_TRUE(( grid.ll_map( cells(), vertexes(), gridtools::static_int<1>(), {2,5}) == array<uint_t, 3>{24*d3, 32*d3, 33*d3} ));

}

TEST_P(ll_map_test, edge_to_edge) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // edge # 51
    ASSERT_TRUE(( grid.ll_map( edges(), edges(), gridtools::static_int<0>(), {2,3}) == array<uint_t, 4>{66*d3, 59*d3, 67*d3, 82*d3} ));
    // edge # 59
    ASSERT_TRUE(( grid.ll_map( edges(), edges(), gridtools::static_int<1>(), {2,3}) == array<uint_t, 4>{43*d3, 28*d3, 67*d3, 51*d3} ));
    // edge # 67
    ASSERT_TRUE(( grid.ll_map( edges(), edges(), gridtools::static_int<2>(), {2,3}) == array<uint_t, 4>{51*d3, 59*d3, 52*d3, 83*d3} ));

    // edge # 123
    ASSERT_TRUE(( grid.ll_map( edges(), edges(), gridtools::static_int<0>(), {5,3}) == array<uint_t, 4>{138*d3, 131*d3, 139*d3, 154*d3} ));
    // edge # 131
    ASSERT_TRUE(( grid.ll_map( edges(), edges(), gridtools::static_int<1>(), {5,3}) == array<uint_t, 4>{115*d3, 100*d3, 139*d3, 123*d3} ));
    // edge # 139
    ASSERT_TRUE(( grid.ll_map( edges(), edges(), gridtools::static_int<2>(), {5,3}) == array<uint_t, 4>{123*d3, 131*d3, 124*d3, 155*d3} ));

}

TEST_P(ll_map_test, edge_to_cell) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // edge # 51
    ASSERT_TRUE(( grid.ll_map( edges(), cells(), gridtools::static_int<0>(), {2,3}) ==
                  array<uint_t, 2>{42*d3, 35*d3} ));
    // edge # 59
    ASSERT_TRUE(( grid.ll_map( edges(), cells(), gridtools::static_int<1>(), {2,3}) ==
                  array<uint_t, 2>{27*d3, 35*d3} ));
    // edge # 67
    ASSERT_TRUE(( grid.ll_map( edges(), cells(), gridtools::static_int<2>(), {2,3}) ==
                  array<uint_t, 2>{35*d3, 43*d3} ));

    // edge # 123
    ASSERT_TRUE(( grid.ll_map( edges(), cells(), gridtools::static_int<0>(), {5,3}) ==
                  array<uint_t, 2>{90*d3, 83*d3} ));
    // edge # 131
    ASSERT_TRUE(( grid.ll_map( edges(), cells(), gridtools::static_int<1>(), {5,3}) ==
                  array<uint_t, 2>{75*d3, 83*d3} ));
    // edge # 139
    ASSERT_TRUE(( grid.ll_map( edges(), cells(), gridtools::static_int<2>(), {5,3}) ==
                  array<uint_t, 2>{83*d3, 91*d3} ));

}

TEST_P(ll_map_test, edge_to_vertex) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // edge # 51
    ASSERT_TRUE(( grid.ll_map( edges(), vertexes(), gridtools::static_int<0>(), {2,3}) ==
                  array<uint_t, 4>{21*d3, 22*d3, 30*d3, 29*d3} ));
    // edge # 59
    ASSERT_TRUE(( grid.ll_map( edges(), vertexes(), gridtools::static_int<1>(), {2,3}) ==
                  array<uint_t, 4>{21*d3, 13*d3, 22*d3, 30*d3} ));
    // edge # 67
    ASSERT_TRUE(( grid.ll_map( edges(), vertexes(), gridtools::static_int<2>(), {2,3}) ==
                  array<uint_t, 4>{21*d3, 22*d3, 30*d3, 31*d3} ));

    // edge # 123
    ASSERT_TRUE(( grid.ll_map( edges(), vertexes(), gridtools::static_int<0>(), {5,3}) ==
                  array<uint_t, 4>{48*d3, 49*d3, 56*d3, 57*d3} ));
    // edge # 131
    ASSERT_TRUE(( grid.ll_map( edges(), vertexes(), gridtools::static_int<1>(), {5,3}) ==
                  array<uint_t, 4>{48*d3, 40*d3, 49*d3, 57*d3} ));
    // edge # 139
    ASSERT_TRUE(( grid.ll_map( edges(), vertexes(), gridtools::static_int<2>(), {5,3}) ==
                  array<uint_t, 4>{48*d3, 49*d3, 57*d3, 58*d3} ));

}

TEST_P(ll_map_test, vertex_to_vertex) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // vertex # 21
    ASSERT_TRUE(( grid.ll_map( vertexes(), vertexes(), gridtools::static_int<0>(), {2,3}) ==
                  array<uint_t, 6>{20*d3, 12*d3, 13*d3, 22*d3, 30*d3, 29*d3} ));
    // vertex # 48
    ASSERT_TRUE(( grid.ll_map( vertexes(), vertexes(), gridtools::static_int<0>(), {5,3}) ==
                  array<uint_t, 6>{39*d3, 40*d3, 49*d3, 57*d3, 56*d3, 47*d3} ));
    // vertex # 60
    ASSERT_TRUE(( grid.ll_map( vertexes(), vertexes(), gridtools::static_int<0>(), {6,6}) ==
                  array<uint_t, 6>{59*d3, 51*d3, 52*d3, 61*d3, 69*d3, 68*d3} ));

}

TEST_P(ll_map_test, vertex_to_cells) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // vertex # 21
    ASSERT_TRUE(( grid.ll_map( vertexes(), cells(), gridtools::static_int<0>(), {2,3}) ==
                  array<uint_t, 6>{26*d3, 19*d3, 27*d3, 35*d3, 42*d3, 34*d3} ));
    // vertex # 48
    ASSERT_TRUE(( grid.ll_map( vertexes(), cells(), gridtools::static_int<0>(), {5,3}) ==
                  array<uint_t, 6>{74*d3, 67*d3, 75*d3, 83*d3, 90*d3, 82*d3} ));
    // vertex # 60
    ASSERT_TRUE(( grid.ll_map( vertexes(), cells(), gridtools::static_int<0>(), {6,6}) ==
                  array<uint_t, 6>{93*d3, 86*d3, 94*d3, 101*d3, 102*d3, 109*d3} ));

}

TEST_P(ll_map_test, vertex_to_edges) {
    const uint_t d3=GetParam();
    trapezoid_2D grid( 6, 12, d3 );

    // vertex # 21
    ASSERT_TRUE(( grid.ll_map( vertexes(), edges(), gridtools::static_int<0>(), {2,3}) ==
                  array<uint_t, 6>{58*d3, 27*d3, 43*d3, 59*d3, 51*d3, 66*d3} ));
    // vertex # 48
    ASSERT_TRUE(( grid.ll_map( vertexes(), edges(), gridtools::static_int<0>(), {5,3}) ==
                  array<uint_t, 6>{130*d3, 99*d3, 115*d3, 131*d3, 123*d3, 138*d3} ));
    // vertex # 60
    ASSERT_TRUE(( grid.ll_map( vertexes(), edges(), gridtools::static_int<0>(), {6,6}) ==
                  array<uint_t, 6>{157*d3, 126*d3, 142*d3, 158*d3, 150*d3, 165*d3} ));

}

INSTANTIATE_TEST_CASE_P(InstantiationName,
       ll_map_test,
       ::testing::Values(2, 5, 9));
