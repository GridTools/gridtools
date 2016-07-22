#pragma once


namespace gdl {

    /**
      @class index pair to global position index translation struct

        The index method return the position index of a given element
        in a vector resulting from matrix serialization, given the pair
        of 3D index set in matrix-like form access

      @tparam first dimension length
      @tparam second dimension length
      @tparam third dimension length
     */
    template <uint_t Size0,uint_t Size1,uint_t Size2 = 0>
    struct pair_indexing
    {
        // TODO: generic interfaces required including 2D case
        constexpr inline uint_t index(uint_t i_i1, uint_t i_j1, uint_t i_k1, uint_t i_i2, uint_t i_j2, uint_t i_k2)
        {
            return m_single_indexing.index(i_i1,i_j1,i_k1)*s_total_size + m_single_indexing.index(i_i2,i_j2,i_k2);
        }

        template <uint_t DIR>
        constexpr inline uint_t dims(void) {  return m_single_indexing.template dim<DIR>(); }

        constexpr static uint_t s_total_size{Size0*Size1*Size2};

        // TODO: this should be static constrexpr!
        // TODO: remove hard coded layout
        const gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false> m_single_indexing{Size0,Size1,Size2};
    };

}
