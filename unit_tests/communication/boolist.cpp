/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <common/boollist.hpp>
#include <common/layout_map.hpp>
#include <iostream>
#include <stdlib.h>

int main() {

  bool pass=true;

  for (int i=0; i<100000; ++i) {
    bool v0 = rand()%2, v1 = rand()%2, v2 = rand()%2;

    gridtools::boollist<3> bl1(v0, v1, v2);

    if (bl1.value(0) != v0)
      pass=false;

    if (bl1.value(1) != v1)
      pass=false;

    if (bl1.value(2) != v2)
      pass=false;

    gridtools::boollist<3> bl2=bl1.permute<gridtools::layout_map<1,2,0> >();

    if (bl2.value(0) != v2)
      pass=false;

    if (bl2.value(1) != v0)
      pass=false;

    if (bl2.value(2) != v1)
      pass=false;

    gridtools::boollist<3> bl3=bl1.permute<gridtools::layout_map<2,1,0> >();

    if (bl3.value(0) != v2)
      pass=false;

    if (bl3.value(1) != v1)
      pass=false;

    if (bl3.value(2) != v0)
      pass=false;

    gridtools::boollist<3> bl4=bl1.permute<gridtools::layout_map<0,1,2> >();

    if (bl4.value(0) != v0)
      pass=false;

    if (bl4.value(1) != v1)
      pass=false;

    if (bl4.value(2) != v2)
      pass=false;

  }

  if (pass)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
  return 0;
}
