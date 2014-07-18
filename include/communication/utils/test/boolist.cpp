#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <utils/boollist.h>
#include <utils/layout_map.h>
#include <iostream>
#include <stdlib.h>

int main() {

  bool pass=true;

  for (int i=0; i<100000; ++i) {
    bool v0 = rand()%2, v1 = rand()%2, v2 = rand()%2;

    GCL::gcl_utils::boollist<3> bl1(v0, v1, v2);

    if (bl1.value0 != v0)
      pass=false;
  
    if (bl1.value1 != v1)
      pass=false;

    if (bl1.value2 != v2)
      pass=false;

    GCL::gcl_utils::boollist<3> bl2=bl1.permute<GCL::layout_map<1,2,0> >();

    if (bl2.value0 != v2)
      pass=false;
  
    if (bl2.value1 != v0)
      pass=false;

    if (bl2.value2 != v1)
      pass=false;

    GCL::gcl_utils::boollist<3> bl3=bl1.permute<GCL::layout_map<2,1,0> >();

    if (bl3.value0 != v2)
      pass=false;
  
    if (bl3.value1 != v1)
      pass=false;

    if (bl3.value2 != v0)
      pass=false;

    GCL::gcl_utils::boollist<3> bl4=bl1.permute<GCL::layout_map<0,1,2> >();

    if (bl4.value0 != v0)
      pass=false;
  
    if (bl4.value1 != v1)
      pass=false;

    if (bl4.value2 != v2)
      pass=false;

  }

  if (pass)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
  return 0;
}
