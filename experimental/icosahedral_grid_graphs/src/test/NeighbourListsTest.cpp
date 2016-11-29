/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#define BOOST_TEST_MODULE NeighbourListsUnittest

#include "../NeighbourLists.hpp"
#include "../triangular_2D.hpp"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeighbourListsSuite)

BOOST_AUTO_TEST_CASE(DomainPositionsUnittest) {
    int n = 6;
    int m = 6;
    int triangOffset = m * 2 + 3;

    triangular_storage< triangular_offsets > storage1(
        std::vector< double >(), n, m, 1, triangular_offsets(triangOffset));

    triangular_storage< triangular_offsets > storage2(
        std::vector< double >(), n, m, 2, triangular_offsets(triangOffset));

    triangular_storage< triangular_offsets > storage3(
        std::vector< double >(), n, m, 3, triangular_offsets(triangOffset));

    triangular_storage< triangular_offsets > storage4(
        std::vector< double >(), n, m, 4, triangular_offsets(triangOffset));

    BOOST_CHECK(storage4.StartComputationDomain() == 113);
    BOOST_CHECK(storage3.StartComputationDomain() == 73);
    BOOST_CHECK(storage2.StartComputationDomain() == 41);
    BOOST_CHECK(storage1.StartComputationDomain() == 17);

    BOOST_CHECK(storage1.EndComputationDomain() == 109);
    BOOST_CHECK(storage2.EndComputationDomain() == 153);
    BOOST_CHECK(storage3.EndComputationDomain() == 205);
    BOOST_CHECK(storage4.EndComputationDomain() == 265);

    BOOST_CHECK(storage4.CellInComputeDomain(113));
    BOOST_CHECK(storage4.CellInComputeDomain(264));
    BOOST_CHECK(storage4.CellInComputeDomain(225));
    BOOST_CHECK(storage4.CellInComputeDomain(208));
    BOOST_CHECK(!storage4.CellInComputeDomain(139));
    BOOST_CHECK(!storage4.CellInComputeDomain(181));
    BOOST_CHECK(!storage4.CellInComputeDomain(252));
    BOOST_CHECK(!storage4.CellInComputeDomain(92));
    BOOST_CHECK(!storage4.CellInComputeDomain(85));
    BOOST_CHECK(!storage4.CellInComputeDomain(98));
    BOOST_CHECK(!storage4.CellInComputeDomain(279));
    BOOST_CHECK(!storage4.CellInComputeDomain(280));
    BOOST_CHECK(!storage4.CellInComputeDomain(291));
    BOOST_CHECK(!storage4.CellInComputeDomain(293));

    BOOST_CHECK(storage4.RowId(114) == 0);
    BOOST_CHECK(storage4.ColumnId(114) == 0);
    BOOST_CHECK(storage4.RowId(197) == 3);
    BOOST_CHECK(storage4.ColumnId(197) == 0);
    BOOST_CHECK(storage4.RowId(201) == 3);
    BOOST_CHECK(storage4.ColumnId(201) == 2);
    BOOST_CHECK(storage4.RowId(151) == 1);
    BOOST_CHECK(storage4.ColumnId(151) == 5);
    BOOST_CHECK(storage4.RowId(234) == 4);
    BOOST_CHECK(storage4.ColumnId(234) == 4);
}

// this test checks the list of exceptional cells
BOOST_AUTO_TEST_CASE(BuildExceptionalCells) {
    int n = 6;
    int m = 6;
    int triangOffset = m * 2 + 3;

    std::vector< std::list< int > > neighbourList;
    build_neighbour_list< 4, 6, 6 >(neighbourList);

    // neighbour depth 1
    {
        triangular_storage< triangular_offsets > storage2(
            std::vector< double >(), n, m, 2, triangular_offsets(triangOffset));
        auto exceptCells = build_exceptional_cells(storage2, neighbourList, 1);
        BOOST_CHECK(exceptCells.size() == 0);
    }
    // neighbour depth 2
    {
        triangular_storage< triangular_offsets > storage2(
            std::vector< double >(), n, m, 2, triangular_offsets(triangOffset));
        std::vector< int > exp = {52, 142};
        auto exceptCells = build_exceptional_cells(storage2, neighbourList, 2);
        BOOST_CHECK(exceptCells.size() == 2);
        for (int i = 0; i < exp.size(); ++i) {
            BOOST_CHECK(std::find(exceptCells.begin(), exceptCells.end(), exp[i]) != exceptCells.end());
        }
    }
    // neighbour depth 3
    {
        triangular_storage< triangular_offsets > storage4(
            std::vector< double >(), n, m, 4, triangular_offsets(triangOffset));

        std::vector< int > exp = {254, 253, 255, 124, 123, 151, 264, 113};
        auto exceptCells = build_exceptional_cells(storage4, neighbourList, 3);
        BOOST_CHECK(exceptCells.size() == 8);

        for (int i = 0; i < exp.size(); ++i) {
            BOOST_CHECK(std::find(exceptCells.begin(), exceptCells.end(), exp[i]) != exceptCells.end());
        }
    }
    // neighbour depth 4
    {
        triangular_storage< triangular_offsets > storage4(
            std::vector< double >(), n, m, 4, triangular_offsets(triangOffset));

        std::vector< int > exp = {254, 253, 255, 226, 228, 256, 124, 123, 151, 122, 150, 152, 264, 263, 113, 114};
        auto exceptCells = build_exceptional_cells(storage4, neighbourList, 4);
        //        for(int i=0; i < exceptCells.size(); ++i)
        //            std::cout << "C@ " << exceptCells[i] << std::endl;
        BOOST_CHECK(exceptCells.size() == exp.size());

        for (int i = 0; i < exp.size(); ++i) {
            BOOST_CHECK(std::find(exceptCells.begin(), exceptCells.end(), exp[i]) != exceptCells.end());
        }
    }

    // neighbour depth 5
    {
        triangular_storage< triangular_offsets > storage4(
            std::vector< double >(), n, m, 4, triangular_offsets(triangOffset));

        std::vector< int > exp = {253,
            254,
            255,
            256,
            257,
            225,
            226,
            227,
            228,
            229,
            121,
            122,
            123,
            124,
            141,
            149,
            150,
            151,
            152,
            177,
            179,
            262,
            263,
            264,
            236,
            113,
            114,
            115};
        auto exceptCells = build_exceptional_cells(storage4, neighbourList, 5);
        BOOST_CHECK(exceptCells.size() == exp.size());

        for (int i = 0; i < exp.size(); ++i) {
            BOOST_CHECK(std::find(exceptCells.begin(), exceptCells.end(), exp[i]) != exceptCells.end());
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
