#ifndef _BOOLLIST_H_
#define _BOOLLIST_H_

#include <boost/static_assert.hpp>
#include "defs.hpp"
#include <gridtools.hpp>
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
    template <ushort_t I>
    struct boollist
    {
      static const ushort_t size=I;

    private:
        // const
        bool m_value[I];

    public:

        GT_FUNCTION
        constexpr ushort_t const& size() const{return m_size;}

        GT_FUNCTION
        constexpr bool const& value(ushort_t const& id) const{return m_value[id];}

        boollist(bool v0)
#ifdef CXX11_ENABLED
            :m_value{v0}{}
#else
            {
                m_value[0]=v0;
            }
#endif

        boollist(bool v0, bool v1)
#ifdef CXX11_ENABLED
            :m_value{v0,v1}{}
#else
            {
                m_value[0]=v0;
                m_value[1]=v1;
            }
#endif

        boollist(bool v0, bool v1, bool v2)
#ifdef CXX11_ENABLED
            :m_value{v0,v1,v2}{}
#else
            {
                m_value[0]=v0;
                m_value[1]=v1;
                m_value[2]=v2;
            }
#endif

        boollist(boollist const& bl)
#ifdef CXX11_ENABLED
            :m_value{bl.m_value[0], bl.m_value[1]} //TODO: generalize to arbitrary dimension
            {}
#else
            {
                for (ushort_t i=0; i<I; ++i)
                    m_value[i]=bl.m_value[i];
            }
#endif

      void copy_out(bool *arr) const {
                for (ushort_t i=0; i<I; ++i)
                    arr[i]=m_value[i];
      }

        template <typename LayoutMap>
        boollist<LayoutMap::length> permute(typename boost::enable_if_c<LayoutMap::length==1>::type* a=0) const {
            return boollist<LayoutMap::length>(LayoutMap::template find<0>(m_value));
      }

        template <typename LayoutMap>
        boollist<LayoutMap::length> permute(typename boost::enable_if_c<LayoutMap::length==2>::type* a=0 ) const {
            return boollist<LayoutMap::length>(LayoutMap::template find<0>(m_value), LayoutMap::template find<1>(m_value));
      }

        template <typename LayoutMap>
        boollist<LayoutMap::length> permute(typename boost::enable_if_c<LayoutMap::length==3>::type* a=0 ) const {
            return boollist<LayoutMap::length>(LayoutMap::template find<0>(m_value), LayoutMap::template find<1>(m_value), LayoutMap::template find<2>(m_value));
      }
    };

}

#endif
