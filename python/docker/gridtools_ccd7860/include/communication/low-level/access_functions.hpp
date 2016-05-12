#ifndef _ACCESS_FUNCTIONS_H_
#define _ACCESS_FUNCTIONS_H_

#include "../../common/array.hpp"

namespace gridtools {
  namespace _gcl_internal {

    template <typename T>
    inline int access(gridtools::array<T,2> const& index,
                      gridtools::array<T,2> const& size)
    {
      return index[0]+index[1]*size[0];
    }

    template <typename T>
    inline int access(gridtools::array<T,3> const& index,
                      gridtools::array<T,3> const& size)
    {
      return index[0]+index[1]*size[0] + index[2]*size[0]*size[1];
    }
  }

}
#endif
