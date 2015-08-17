#pragma once
#include "../../include/storage/base_storage_impl.hpp"
#include "../../include/common/layout_map.hpp"
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/at_c.hpp>


namespace gridtools {

    /**
       @brief main class for the basic storage

       The virtual_storage class contains one snapshot. It univocally defines
       the access pattern with three integers: the total storage sizes and
       the two strides different from one.
    */
    template < typename Layout >
    struct virtual_storage
    {
        typedef Layout layout;

        static const ushort_t space_dimensions = layout::length;

    protected:
        array<uint_t, space_dimensions> m_dims;
        //uint_t m_strides[space_dimensions];
        array<uint_t, space_dimensions-1> m_strides;

    private:
        /**@brief noone calls the empty constructor*/
        virtual_storage();

    public:

//         template <typename T, typename U, bool B>
//         friend std::ostream& operator<<(std::ostream &, virtual_storage<T,U, B> const & );

        template <typename Int>
        virtual_storage(  array<Int, space_dimensions> const& sizes ):
            m_dims{sizes},
            m_strides{}
            {
                for (int j=1; j<space_dimensions; ++j) {
                    m_strides[j-1] = 1;
                    for (int i=j; i<space_dimensions; ++i) {
                        m_strides[j-1] *= sizes[i];
                    }
                }
            }

        /**@brief destructor: frees the pointers to the data fields which are not managed outside */
        virtual ~virtual_storage() {}

        /**@brief returns the size of the data field*/
        GT_FUNCTION
        uint_t const& size() const {
            return m_strides[0];
        }

        array<int, space_dimensions> offset2indices(uint_t offset) const {
            array<int, space_dimensions> indices;
            for (int i = 0; i < space_dimensions-1; ++i) {
                indices[i] = offset/m_strides[i];
                offset = offset%m_strides[i];
            }
            indices[space_dimensions-1] = offset;
            return indices;
        }

        /**@brief return the stride for a specific coordinate, given the vector of strides
           Coordinates 0,1,2 correspond to i,j,k respectively*/
        template<uint_t Coordinate>
        GT_FUNCTION
        static constexpr uint_t strides(uint_t const* str)
        {
            return (vec_max<typename layout::layout_vector_t>::value < 0) ?0:( layout::template at_<Coordinate>::value == vec_max<typename layout::layout_vector_t>::value ) ? 1 :  str[layout::template at_<Coordinate>::value];
        }

        /**@brief return the stride for a specific coordinate, given the vector of strides
           Coordinates 0,1,2 correspond to i,j,k respectively*/
        template<uint_t Coordinate>
        GT_FUNCTION
        constexpr uint_t strides() const
        {
            return (vec_max<typename layout::layout_vector_t>::value < 0) ?0:( layout::template at_<Coordinate>::value == vec_max<typename layout::layout_vector_t>::value ) ? 1 :  m_strides[layout::template at_<Coordinate>::value];
        }


        /**
           @brief computing index to access the storage relative to the coordinates passed as parameters.

           This interface must be used with unsigned integers of type uint_t, and the result must be a positive integer as well
        */
        template <typename ... Int>
        GT_FUNCTION
        int_t _index( Int const& ... dims) const {
#ifndef __CUDACC__
            typedef boost::mpl::vector<Int...> tlist;
            //boost::is_same<boost::mpl::_1, uint_t>
            typedef typename boost::mpl::find_if<tlist, boost::mpl::not_< boost::is_integral<boost::mpl::_1> > >::type iter;
            GRIDTOOLS_STATIC_ASSERT(iter::pos::value==sizeof...(Int), "you have to pass in arguments of integral type");
#endif
            return _impl::compute_offset<space_dimensions, layout>::apply(&m_strides[0], dims ...);
        }

        /** @brief returns the memory access index of the element with coordinate (i,j,k) */
        //note: returns a signed int because it might be negative (used e.g. in iterate_domain)
        template<typename IntType>
        GT_FUNCTION
        int_t _index(IntType* indices) const {

            return  _impl::compute_offset<space_dimensions, layout>::apply(m_strides, indices);
        }

        /** @brief returns the dimension fo the field along I*/
        template<ushort_t I>
        GT_FUNCTION
        uint_t dims() const {return m_dims[I];}

        /**@brief returns the storage strides*/
        GT_FUNCTION
        uint_t strides(ushort_t i) const {
            //"you might thing that with m_srides[0] you are accessing a stride,\n
            // but you are indeed accessing the whole storage dimension."
            assert(i!=0);
            return m_strides[i];
        }

    };

} //namespace gridtools
