#ifndef _BOOLLIST_H_
#define _BOOLLIST_H_

#include <boost/static_assert.hpp>
#include "defs.h"
#include <boost/utility/enable_if.hpp>
/*@file
@bief  The following class describes a boolean list of length N.

*/
namespace gridtools {

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


    // template <ushort_t I>
    // struct boollist {}; // brackets to get it into documentation

    template <ushort_t I>
    struct boollist
    {
      static const ushort_t size=I;

    private:
        const bool m_value[I];

    public:

        constexpr bool const& value(ushort_t const& id) const{return m_value[id];}

        boollist(bool v0)
            :m_value{v0}
            {}

        boollist(bool v0, bool v1)
            :m_value{v0,v1}
            {}

        boollist(bool v0, bool v1, bool v2)
            :m_value{v0,v1,v2}
            {}

        boollist(boollist const& bl)
            :m_value{bl.m_value[0], bl.m_value[1]} //TODO: generalize to arbitrary dimension
            {
                // for (ushort_t i=0; i<I; ++i)
                //     m_value[i]=bl.m_value[i];
            }

      template <typename layoutmap>
      boollist<I> permute() const {
          BOOST_STATIC_ASSERT( layoutmap::length == I );
        return boollist<I>(layoutmap::template find<0>(m_value));
      }

      void copy_out(bool *arr) const {
                for (ushort_t i=0; i<I; ++i)
                    arr[i]=m_value[i];
      }

        template <typename LayoutMap>
        boollist<LayoutMap::length> permute(typename boost::enable_if_c<LayoutMap::length==1>::type a=0) const {
            return boollist<LayoutMap::length>(LayoutMap::template find<0>(m_value));
      }

        template <typename LayoutMap>
        boollist<LayoutMap::length> permute(typename boost::enable_if_c<LayoutMap::length==2>::type a=0 ) const {
            return boollist<LayoutMap::length>(LayoutMap::template find<0>(m_value), LayoutMap::template find<1>(m_value));
      }

        template <typename LayoutMap>
        boollist<LayoutMap::length> permute(typename boost::enable_if_c<LayoutMap::length==3>::type a=0 ) const {
            return boollist<LayoutMap::length>(LayoutMap::template find<0>(m_value), LayoutMap::template find<1>(m_value), LayoutMap::template find<2>(m_value));
      }
    };



    // template <>
    // struct boollist<2>
    // {
    //   static const ushort_t size=2;
    //   const bool value0;
    //   const bool value1;
    //   boollist(bool v0, bool v1)
    //     : value0(v0)
    //     , value1(v1)
    //   {}
    //   boollist(boollist const& bl)
    //     :value0(bl.value0)
    //     ,value1(bl.value1)
    //   {}


    //   template <typename layoutmap>
    //   boollist<2> permute() const {
    //     BOOST_STATIC_ASSERT( layoutmap::length == 2 );
    //     return boollist<2>(layoutmap::template find<0>(value0, value1),
    //                        layoutmap::template find<1>(value0, value1));
    //   }

    //   void copy_out(bool *arr) const {
    //         arr[0] = value0;
    //         arr[1] = value1;
    //   }
    // };

    // template <>
    // struct boollist<3>
    // {
    //   static const ushort_t size=3;
    //   const bool value0;
    //   const bool value1;
    //   const bool value2;
    //   boollist(bool v0, bool v1, bool v2)
    //     : value0(v0)
    //     , value1(v1)
    //     , value2(v2)
    //   {}
    //   boollist(boollist const& bl)
    //     : value0(bl.value0)
    //     , value1(bl.value1)
    //     , value2(bl.value2)
    //   {}

    //   template <typename layoutmap>
    //   boollist<3> permute() const {
    //     BOOST_STATIC_ASSERT( layoutmap::length == 3 );
    //     return boollist<3>(layoutmap::template find<0>(value0, value1, value2),
    //                        layoutmap::template find<1>(value0, value1, value2),
    //                        layoutmap::template find<2>(value0, value1, value2));
    //   }

    //   void copy_out(bool *arr) const {
    //         arr[0] = value0;
    //         arr[1] = value1;
    //         arr[2] = value2;
    //   }
    // };

    // template <>
    // struct boollist<4>
    // {
    //   static const ushort_t size=4;
    //   const bool value0;
    //   const bool value1;
    //   const bool value2;
    //   const bool value3;
    //   boollist(bool v0, bool v1, bool v2, bool v3)
    //     :value0(v0)
    //     , value1(v1)
    //     , value2(v2)
    //     , value3(v3)
    //   {}
    //   boollist(boollist const& bl)
    //     : value0(bl.value0)
    //     , value1(bl.value1)
    //     , value2(bl.value2)
    //     , value3(bl.value3)
    //   {}

    //   template <typename layoutmap>
    //   boollist<4> permute() const {
    //     BOOST_STATIC_ASSERT( layoutmap::length == 4 );
    //     return boollist<4>(layoutmap::template find<0>(value0, value1, value2, value3),
    //                        layoutmap::template find<1>(value0, value1, value2, value3),
    //                        layoutmap::template find<2>(value0, value1, value2, value3),
    //                        layoutmap::template find<3>(value0, value1, value2, value3));
    //   }

    //   void copy_out(bool *arr) const {
    //         arr[0] = value0;
    //         arr[1] = value1;
    //         arr[2] = value2;
    //         arr[3] = value3;
    //   }
    // };

}

#endif
