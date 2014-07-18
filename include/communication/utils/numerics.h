#ifndef _NUMERICS_H_
#define _NUMERICS_H_

namespace GCL {
  namespace _impl {
    // Compute 3^I at compile time
    template <int I>
    struct static_pow3;

    template <>
    struct static_pow3<1> {
      static const int value = 3;
    };

    template <int I>
    struct static_pow3 {
      static const int value = 3*static_pow3<I-1>::value;
    };

  }
}

#endif
