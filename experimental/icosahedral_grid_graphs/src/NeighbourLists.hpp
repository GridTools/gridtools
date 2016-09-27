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
/*
 * NeighbourLists.h
 *
 *  Created on: Oct 30, 2014
 *      Author: carlosos
 */

#pragma once
#include <map>
#include "triangular_2D.hpp"

template<int VHaloSize>
void insert_middle_row(std::vector<std::list<int> >& neighbours, int cellStart, int cellEnd, int triang_offset, int sign_even_stride)
{
    for(int i=cellStart; i < cellEnd; ++i) {
        if(i%2!=0) {
            neighbours.push_back({i-1,i+1,i-triang_offset*sign_even_stride});
        }
        else{
            neighbours.push_back({i-1,i+1,i+triang_offset*sign_even_stride});
        }
    }

}
void insert_padding(std::vector<std::list<int> >& neighbours, int nPaddedCells)
{
    for(int i=0; i < nPaddedCells; ++i)
    {
        neighbours.push_back({});
    }
}

// build a list of neighbours per cells for certain domain configuration
// Each line described by VHaloSize, VNRows or VNColumns is composed by a diamond
// (i.e. two triangular cells)
template<int VHaloSize, int VNRows, int VNColumns>
void build_neighbour_list(std::vector<std::list<int> >& neighbours);

// configuration with 1 halo line and 6x6 compute domain
template<> void build_neighbour_list<1,6,6>(std::vector<std::list<int> >& neighbours) {

    int m=6;
    int n=6;
    int triang_offset=m*2+3;
    int jstride=triang_offset+1;
    int row_padd=3;
    int haloSize = 1;

    neighbours.push_back( { 1, 16 });
    std::cout << "N insert " << 0 << " -> " << 1 << " " << 16 << std::endl;
    neighbours.push_back( { 0, 2 });
    std::cout << "N insert " << 1 << " -> " << 0 << " " << 2 << std::endl;
    neighbours.push_back( { 1, 3, 17 });
    std::cout << "N insert " << 2 << " -> " << 1 << " " << 3 << " " << 17
            << std::endl;
    for (int i = 3; i < 13; ++i) {
        if (i % 2 != 0) {
            std::cout << "N insert " << i << " -> " << i - 1 << " " << i + 1
                    << std::endl;
            neighbours.push_back( { i - 1, i + 1 });
        } else {
            std::cout << "N insert " << i << " -> " << i - 1 << " " << i + 1
                    << " " << triang_offset << std::endl;
            neighbours.push_back( { i - 1, i + 1, i + triang_offset });
        }
    }

    std::cout << "N insert " << 13 << " -> " << 12 << " " << 29 << std::endl;
    neighbours.push_back( { 12, 29 }); // cell id 13
    std::cout << "N insert " << 14 << " -> " << std::endl;
    neighbours.push_back( { }); // cell id 14
    std::cout << "N insert " << 15 << " -> " << std::endl;
    neighbours.push_back( { }); // cell id 15
    std::cout << "N insert " << 16 << " -> " << 0 << " " << 17 << " " << 31
            << std::endl;
    neighbours.push_back( { 0, 17, 31 }); // cell id 16
    for (int i = 17; i < 29; ++i) {
        if (i % 2 == 0) {
            std::cout << "N insert " << i << " -> " << i - 1 << " " << i + 1
                    << " " << i + triang_offset << std::endl;
            neighbours.push_back( { i - 1, i + 1, i + triang_offset });
        } else {
            std::cout << "N insert " << i << " -> " << i - 1 << " " << i + 1
                    << "  " << i - triang_offset << std::endl;
            neighbours.push_back( { i - 1, i + 1, i - triang_offset });

        }
    }
    std::cout << "N insert " << 29 << " -> " << 28 << " " << 30 << " " << 13
            << std::endl;
    neighbours.push_back( { 28, 30, 13 }); // cell id 29
    std::cout << "N insert " << 30 << " -> " << 29 << " " << 45 << std::endl;
    neighbours.push_back( { 29, 45 }); // cell id 30
    for (int j = 0; j < 4; ++j) {
        int firstCell = 31 + j * (jstride);
        neighbours.push_back( { firstCell - triang_offset, firstCell + 1 }); // cell id 31
        for (int i = firstCell + 1; i < firstCell + triang_offset; ++i) { //cell ids [31-45]
            if (i % 2 == 0)
                neighbours.push_back( { i - 1, i + 1, i + triang_offset });
            else
                neighbours.push_back( { i - 1, i + 1, i - triang_offset });
        }
        neighbours.push_back( { firstCell - 1, firstCell + (triang_offset) }); // cell id 46
    }
    neighbours.push_back( { 80, 96 }); // cell id 95
    for (int i = 96; i < 110; ++i) { //cell ids [96-109]
        if (i % 2 == 0)
            neighbours.push_back( { i - 1, i + 1, i + triang_offset });
        else
            neighbours.push_back( { i - 1, i + 1, i - triang_offset });
    }

    neighbours.push_back( { 109, 124 }); // cell id 110
    neighbours.push_back( { 96, 113 }); // cell id 111
    neighbours.push_back( { }); // cell id 112
    neighbours.push_back( { 111, 98, 114 }); // cell id 113
    for (int i = 114; i < 124; ++i) { //cell ids [114-123]
        if (i % 2 == 0) {
            neighbours.push_back( { i - 1, i + i });
        } else {
            neighbours.push_back( { i - 1, i + 1, i - triang_offset });
        }
    }
    neighbours.push_back( { 123, 110 }); // cell id 124
}

// configuration with 4 halo line and 6x6 compute domain
template<> void build_neighbour_list<4,6,6>(std::vector<std::list<int> >& neighbours)
{
    const int HaloSize=8;

    int triang_offset = 27;
    int stride = triang_offset+1;
    //padding
    insert_padding(neighbours, 7);
    neighbours.push_back( {8,32} );  // cell id 7
    for(int i=9; i < 20; ++i) {
        if(i%2==0) {
            neighbours.push_back({i-1,i+1,i+triang_offset});
        }
        else{
            neighbours.push_back({i-1,i+1});
        }
    }
    neighbours.push_back( {18} );  // cell id 19
    insert_padding(neighbours, 11);
    neighbours.push_back( {32,56} );  // cell id 31
    insert_middle_row<HaloSize>(neighbours, 32, 34, triang_offset, -1);
    insert_middle_row<HaloSize>(neighbours, 34, 45, triang_offset, 1);
    neighbours.push_back( {18,44} );  // cell id 45
    insert_padding(neighbours, 9);
    neighbours.push_back( {56,80} );  // cell id 55
    insert_middle_row<HaloSize>(neighbours, 56, 60, triang_offset,-1);
    insert_middle_row<HaloSize>(neighbours, 60, 71, triang_offset,1);
    neighbours.push_back( {70,44} );  // cell id 71
    insert_padding(neighbours, 7);
    neighbours.push_back( {80,105} );  // cell id 79
    insert_middle_row<HaloSize>(neighbours, 80, 86, triang_offset,-1);
    insert_middle_row<HaloSize>(neighbours, 86, 97, triang_offset,1);
    neighbours.push_back( {96,70} );  // cell id 97
    insert_padding(neighbours, 7);
    neighbours.push_back( {79,106} );  // cell id 105
    insert_middle_row<HaloSize>(neighbours, 106, 125, triang_offset,1);
    neighbours.push_back( {124,126,97} );  // cell id 125
    neighbours.push_back( {125,127,153} );  // cell id 126
    neighbours.push_back( {126,128,71} );  // cell id 127
    neighbours.push_back( {127,129,155} );  // cell id 128
    neighbours.push_back( {128,130,45} );  // cell id 129
    neighbours.push_back( {139,131,157} );  // cell id 130
    neighbours.push_back( {130,132,19} );  // cell id 131
    neighbours.push_back( {131,159} );  // cell id 132

    int startCell=133;
    for(int j=0; j !=4; ++j)
    {
        int firstCellRow = startCell+stride*j;
        int lastCellRow = firstCellRow+stride-1;
        neighbours.push_back( {firstCellRow-triang_offset,firstCellRow+1} );  // cell id 133
        insert_middle_row<HaloSize>(neighbours, firstCellRow+1, lastCellRow, triang_offset,1);
        neighbours.push_back( {lastCellRow,lastCellRow+triang_offset} );  // cell id 160
    }
    neighbours.push_back( {218,246} );  // cell id 245
    insert_middle_row<HaloSize>(neighbours, 246, 266, triang_offset,1);
    neighbours.push_back( {265,267,292} );  // cell id 266
    neighbours.push_back( {266,268,240} );  // cell id 267
    neighbours.push_back( {267,269,320} );  // cell id 268
    neighbours.push_back( {268,270,242} );  // cell id 269
    neighbours.push_back( {269,271,348} );  // cell id 270
    neighbours.push_back( {270,272,244} );  // cell id 271
    neighbours.push_back( {271,376} );  // cell id 272
    neighbours.push_back( {246,274} );  // cell id 273
    insert_middle_row<HaloSize>(neighbours, 274, 279, triang_offset,1);
    neighbours.push_back( {278,281,252} );  // cell id 279
    insert_padding(neighbours,1);
    neighbours.push_back( {279,281,254} );  // cell id 281
    insert_middle_row<HaloSize>(neighbours, 282, 292, triang_offset,1);
    neighbours.push_back( {291,319,266} );  // cell id 292
    insert_padding(neighbours,8);
    neighbours.push_back( {274,302} );  // cell id 301
    insert_middle_row<HaloSize>(neighbours, 302, 305, triang_offset,1);
    neighbours.push_back( {304,278,309} );  // cell id 305
    insert_padding(neighbours,3);
    neighbours.push_back( {305,310,282} );  // cell id 309
    insert_middle_row<HaloSize>(neighbours, 310, 320, triang_offset,1);
    neighbours.push_back( {319,347,268} );  // cell id 320
    insert_padding(neighbours,8);
    neighbours.push_back( {302,330} );  // cell id 329
    neighbours.push_back( {329,331,357} );  // cell id 330
    neighbours.push_back( {330,304,337} );  // cell id 331
    insert_padding(neighbours, 5);
    neighbours.push_back( {331,310,338} );  // cell id 337
    insert_middle_row<HaloSize>(neighbours, 338, 348, triang_offset,1);
    neighbours.push_back( {347,375,270} );  // cell id 348
    insert_padding(neighbours,8);
    neighbours.push_back( {330,365} );  // cell id 357
    insert_padding(neighbours,7);
    neighbours.push_back( {357,338,366} );  // cell id 365

    for(int i=366; i < 376; ++i) {
        if(i%2==0) {
            neighbours.push_back({i-1,i+1});
        }
        else{
            neighbours.push_back({i-1,i+1,i-triang_offset});
        }
    }
    neighbours.push_back( {375,272} );  // cell id 376
    insert_padding(neighbours,8);

}

void appendNeighboursInComputeDomain(const triangular_storage<triangular_offsets>& storage, std::vector<int>& exceptionalCells, std::map<int,bool>& insertedCells,
        int cellId, std::vector<std::list<int> > neighbourList, int neighbourDepth)
{
    if(neighbourDepth==0) return;

    for(auto iter = neighbourList[cellId].begin(); iter != neighbourList[cellId].end(); ++iter)
    {
        if(!insertedCells.count(*iter)) {
            if(storage.CellInComputeDomain(*iter)) {
                insertedCells[*iter] = true;
                exceptionalCells.push_back(*iter);
            }
            int neighbourId = *iter;
            appendNeighboursInComputeDomain(storage, exceptionalCells, insertedCells, neighbourId, neighbourList, neighbourDepth-1);
        }
    }
}

std::vector<int> build_exceptional_cells(const triangular_storage<triangular_offsets>& storage, const std::vector<std::list<int> >& neighbours, int neighbourDepth)
{
    if(neighbourDepth==1) return std::vector<int>();

    int startDomain = storage.StartComputationDomain();
    int endDomain = storage.EndComputationDomain();
    std::vector<int> exceptionalCells;
    std::map<int,bool> insertedCells;
    // Add corner cell (in computation domain) between diamond 2 and 6
    int cellId = startDomain+storage.nColumns*2-1;
    exceptionalCells.push_back(cellId);
    insertedCells[cellId]=true;
    cellId = endDomain-storage.nColumns*2+1;
    exceptionalCells.push_back(cellId);
    insertedCells[cellId]=true;


    if(neighbourDepth==2)
        return exceptionalCells;
    auto inputCells = exceptionalCells;
    for(auto iter = inputCells.begin(); iter != inputCells.end(); ++iter)
    {
        appendNeighboursInComputeDomain(storage, exceptionalCells,insertedCells, *iter, neighbours, 1);
    }
    exceptionalCells.push_back(startDomain);
    insertedCells[startDomain]=true;
    exceptionalCells.push_back(endDomain-1);
    insertedCells[endDomain-1]=true;

    inputCells = exceptionalCells;
    for(auto iter = inputCells.begin(); iter != inputCells.end(); ++iter)
    {
        appendNeighboursInComputeDomain(storage, exceptionalCells,insertedCells, *iter, neighbours, neighbourDepth-3);
    }
    return exceptionalCells;
}
