#pragma once
namespace gridtools{

    /**
       @brief returns the padding to be added at the end of a specific dimension

       used internally in the library to allocate the storages
     */
    template <ushort_t AlignmentBoundary>
    constexpr uint_t align(uint_t const& dimension){
        return AlignmentBoundary ? dimension+AlignmentBoundary-(dimension%AlignmentBoundary) : dimension;
    }

    template<ushort_t Boundary>
    struct aligned{
        static const ushort_t value=Boundary;
    };
}//namespace gridtools
