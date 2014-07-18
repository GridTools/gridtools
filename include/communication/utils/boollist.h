#ifndef _BOOLLIST_H_
#define _BOOLLIST_H_

#include <boost/static_assert.hpp>

namespace GCL {

  /**
     namespace containing utility finctions to be used with GCL
  */
  namespace gcl_utils {

    /**
       The following class describes a boolean list of length N.
       This is used in proc_grids. 

       \code
       boollist<4> bl(true, false, false, true);
       bl.value3 == true
       bl.value2 == false
       \endcode
       See \link Concepts \endlink, \link proc_grid_2D_concept \endlink, \link proc_grid_3D_concept \endlink
     */
    template <int I>
    struct boollist {}; // brackets to get it into documentation

    template <>
    struct boollist<1> 
    {
      const bool value0;
      boollist(bool v0) 
        :value0(v0) 
      {}
      boollist(boollist const& bl)
        :value0(bl.value0) 
      {}

      template <typename layoutmap>
      boollist<1> permute() const {
        BOOST_STATIC_ASSERT( layoutmap::length == 1 );
        return boollist<1>(layoutmap::template find<0>(value0));
      }

      void copy_out(bool *arr) const {
            arr[0] = value0;
      }

    };

    template <>
    struct boollist<2> 
    {
      const bool value0;
      const bool value1;
      boollist(bool v0, bool v1) 
        : value0(v0)
        , value1(v1) 
      {}
      boollist(boollist const& bl)
        :value0(bl.value0) 
        ,value1(bl.value1) 
      {}


      template <typename layoutmap>
      boollist<2> permute() const {
        BOOST_STATIC_ASSERT( layoutmap::length == 2 );
        return boollist<2>(layoutmap::template find<0>(value0, value1),
                           layoutmap::template find<1>(value0, value1));
      }

      void copy_out(bool *arr) const {
            arr[0] = value0;
            arr[1] = value1;
      }
    };

    template <>
    struct boollist<3> 
    {
      const bool value0;
      const bool value1;
      const bool value2;
      boollist(bool v0, bool v1, bool v2) 
        : value0(v0)
        , value1(v1) 
        , value2(v2)
      {}
      boollist(boollist const& bl)
        : value0(bl.value0) 
        , value1(bl.value1) 
        , value2(bl.value2) 
      {}

      template <typename layoutmap>
      boollist<3> permute() const {
        BOOST_STATIC_ASSERT( layoutmap::length == 3 );
        return boollist<3>(layoutmap::template find<0>(value0, value1, value2),
                           layoutmap::template find<1>(value0, value1, value2),
                           layoutmap::template find<2>(value0, value1, value2));
      }

      void copy_out(bool *arr) const {
            arr[0] = value0;
            arr[1] = value1;
            arr[2] = value2;
      }
    };

    template <>
    struct boollist<4> 
    {
      const bool value0;
      const bool value1;
      const bool value2;
      const bool value3;
      boollist(bool v0, bool v1, bool v2, bool v3) 
        :value0(v0)
        , value1(v1) 
        , value2(v2) 
        , value3(v3)
      {}
      boollist(boollist const& bl)
        : value0(bl.value0) 
        , value1(bl.value1) 
        , value2(bl.value2) 
        , value3(bl.value3) 
      {}

      template <typename layoutmap>
      boollist<4> permute() const {
        BOOST_STATIC_ASSERT( layoutmap::length == 4 );
        return boollist<4>(layoutmap::template find<0>(value0, value1, value2, value3),
                           layoutmap::template find<1>(value0, value1, value2, value3),
                           layoutmap::template find<2>(value0, value1, value2, value3),
                           layoutmap::template find<3>(value0, value1, value2, value3));
      }

      void copy_out(bool *arr) const {
            arr[0] = value0;
            arr[1] = value1;
            arr[2] = value2;
            arr[3] = value3;
      }
    };


  }
}


#endif
