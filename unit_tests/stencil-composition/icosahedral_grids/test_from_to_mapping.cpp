#include "gtest/gtest.h"

#include <stencil_composition/stencil_composition.hpp>

using namespace gridtools;

using icosahedral_topology_t =
    gridtools::icosahedral_topology< gridtools::backend< enumtype::Host, enumtype::icosahedral, enumtype::Block > >;

class ll_map_test : public ::testing::TestWithParam<int> {
    // You can implement all the usual fixture class members here.
    // To access the test parameter, call GetParam() from class
    // TestWithParam<T>.
};

TEST_P(ll_map_test, cell_to_cell) {

    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // cell # 50
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), cells(), gridtools::static_int<0>(), {(uint_t)3,(uint_t)2,(uint_t)i}),
                      array<uint_t, 3>{42*d3+i, 58*d3+i, 57*d3+i} ));
        // cell # 58
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), cells(), gridtools::static_int<1>(), {(uint_t)3,(uint_t)2,(uint_t)i}),
                      array<uint_t, 3>{50*d3+i, 51*d3+i, 66*d3+i} ));
        // cell # 35
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), cells(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 3>{42*d3+i, 43*d3+i, 27*d3+i} ));
        // cell # 45
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), cells(), gridtools::static_int<1>(), {(uint_t)2,(uint_t)5,(uint_t)i}),
                      array<uint_t, 3>{53*d3+i, 38*d3+i, 37*d3+i} ));

    }
}

TEST_P(ll_map_test, cell_to_edge) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // cell # 50
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), edges(), gridtools::static_int<0>(), {(uint_t)3,(uint_t)2,(uint_t)i}),
                      array<uint_t, 3>{74*d3+i, 82*d3+i, 90*d3+i} ));
        // cell # 58
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), edges(), gridtools::static_int<1>(), {(uint_t)3,(uint_t)2,(uint_t)i}),
                      array<uint_t, 3>{90*d3+i, 75*d3+i, 106*d3+i} ));
        // cell # 35
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), edges(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 3>{51*d3+i, 59*d3+i, 67*d3+i} ));
        // cell # 45
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), edges(), gridtools::static_int<1>(), {(uint_t)2,(uint_t)5,(uint_t)i}),
                      array<uint_t, 3>{69*d3+i, 54*d3+i, 85*d3+i} ));

    }
}

TEST_P(ll_map_test, cell_to_vertex) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // cell # 50
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), vertexes(), gridtools::static_int<0>(), {(uint_t)3,(uint_t)2,(uint_t)i}),
                      array<uint_t, 3>{29*d3+i, 30*d3+i, 38*d3+i} ));
        // cell # 58
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), vertexes(), gridtools::static_int<1>(), {(uint_t)3,(uint_t)2,(uint_t)i}),
                      array<uint_t, 3>{38*d3+i, 30*d3+i, 39*d3+i} ));
        // cell # 35
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), vertexes(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 3>{21*d3+i, 22*d3+i, 30*d3+i} ));
        // cell # 45
        ASSERT_TRUE(same_elements( grid.ll_map( cells(), vertexes(), gridtools::static_int<1>(), {(uint_t)2,(uint_t)5,(uint_t)i}),
                      array<uint_t, 3>{24*d3+i, 32*d3+i, 33*d3+i} ));

    }
}

TEST_P(ll_map_test, edge_to_edge) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // edge # 51
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), edges(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{66*d3+i, 59*d3+i, 67*d3+i, 82*d3+i} ));
        // edge # 59
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), edges(), gridtools::static_int<1>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{43*d3+i, 28*d3+i, 67*d3+i, 51*d3+i} ));
        // edge # 67
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), edges(), gridtools::static_int<2>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{51*d3+i, 59*d3+i, 52*d3+i, 83*d3+i} ));

        // edge # 123
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), edges(), gridtools::static_int<0>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{138*d3+i, 131*d3+i, 139*d3+i, 154*d3+i} ));
        // edge # 131
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), edges(), gridtools::static_int<1>(), {(uint_t)5,(uint_t)3,(uint_t)i}), 
                      array<uint_t, 4>{115*d3+i, 100*d3+i, 139*d3+i, 123*d3+i} ));
        // edge # 139
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), edges(), gridtools::static_int<2>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{123*d3+i, 131*d3+i, 124*d3+i, 155*d3+i} ));

    }
}

TEST_P(ll_map_test, edge_to_cell) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // edge # 51
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), cells(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 2>{42*d3+i, 35*d3+i} ));
        // edge # 59
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), cells(), gridtools::static_int<1>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 2>{27*d3+i, 35*d3+i} ));
        // edge # 67
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), cells(), gridtools::static_int<2>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 2>{35*d3+i, 43*d3+i} ));

        // edge # 123
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), cells(), gridtools::static_int<0>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 2>{90*d3+i, 83*d3+i} ));
        // edge # 131
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), cells(), gridtools::static_int<1>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 2>{75*d3+i, 83*d3+i} ));
        // edge # 139
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), cells(), gridtools::static_int<2>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                  array<uint_t, 2>{83*d3+i, 91*d3+i} ));

    }
}

TEST_P(ll_map_test, edge_to_vertex) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // edge # 51
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), vertexes(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{21*d3+i, 22*d3+i, 30*d3+i, 29*d3+i} ));
        // edge # 59
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), vertexes(), gridtools::static_int<1>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{21*d3+i, 13*d3+i, 22*d3+i, 30*d3+i} ));
        // edge # 67
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), vertexes(), gridtools::static_int<2>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{21*d3+i, 22*d3+i, 30*d3+i, 31*d3+i} ));

        // edge # 123
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), vertexes(), gridtools::static_int<0>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{48*d3+i, 49*d3+i, 56*d3+i, 57*d3+i} ));
        // edge # 131
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), vertexes(), gridtools::static_int<1>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{48*d3+i, 40*d3+i, 49*d3+i, 57*d3+i} ));
        // edge # 139
        ASSERT_TRUE(same_elements( grid.ll_map( edges(), vertexes(), gridtools::static_int<2>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 4>{48*d3+i, 49*d3+i, 57*d3+i, 58*d3+i} ));

    }
}

TEST_P(ll_map_test, vertex_to_vertex) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // vertex # 21
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), vertexes(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 6>{20*d3+i, 12*d3+i, 13*d3+i, 22*d3+i, 30*d3+i, 29*d3+i} ));
        // vertex # 48
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), vertexes(), gridtools::static_int<0>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 6>{39*d3+i, 40*d3+i, 49*d3+i, 57*d3+i, 56*d3+i, 47*d3+i} ));
        // vertex # 60
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), vertexes(), gridtools::static_int<0>(), {(uint_t)6,(uint_t)6,(uint_t)i}),
                      array<uint_t, 6>{59*d3+i, 51*d3+i, 52*d3+i, 61*d3+i, 69*d3+i, 68*d3+i} ));

    }
}

TEST_P(ll_map_test, vertex_to_cells) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // vertex # 21
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), cells(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 6>{26*d3+i, 19*d3+i, 27*d3+i, 35*d3+i, 42*d3+i, 34*d3+i} ));
        // vertex # 48
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), cells(), gridtools::static_int<0>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 6>{74*d3+i, 67*d3+i, 75*d3+i, 83*d3+i, 90*d3+i, 82*d3+i} ));
        // vertex # 60
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), cells(), gridtools::static_int<0>(), {(uint_t)6,(uint_t)6,(uint_t)i}),
                      array<uint_t, 6>{93*d3+i, 86*d3+i, 94*d3+i, 101*d3+i, 102*d3+i, 109*d3+i} ));

    }
}

TEST_P(ll_map_test, vertex_to_edges) {
    const uint_t d3=GetParam();
    icosahedral_topology_t grid( 8, 8, d3 );

    for (int i = 0; i < d3; i+=2) {
        // vertex # 21
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), edges(), gridtools::static_int<0>(), {(uint_t)2,(uint_t)3,(uint_t)i}),
                      array<uint_t, 6>{58*d3+i, 27*d3+i, 43*d3+i, 59*d3+i, 51*d3+i, 66*d3+i} ));
        // vertex # 48
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), edges(), gridtools::static_int<0>(), {(uint_t)5,(uint_t)3,(uint_t)i}),
                      array<uint_t, 6>{130*d3+i, 99*d3+i, 115*d3+i, 131*d3+i, 123*d3+i, 138*d3+i} ));
        // vertex # 60
        ASSERT_TRUE(same_elements( grid.ll_map( vertexes(), edges(), gridtools::static_int<0>(), {(uint_t)6,(uint_t)6,(uint_t)i}),
                      array<uint_t, 6>{157*d3+i, 126*d3+i, 142*d3+i, 158*d3+i, 150*d3+i, 165*d3+i} ));

    }
}

INSTANTIATE_TEST_CASE_P(InstantiationName,
                        ll_map_test,
                        ::testing::Values(2, 5, 9));
