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

    /**nvcc does not understand that the function above is a constant expression*/
    template <ushort_t AlignmentBoundary, uint_t Dimension>
    struct align_struct{
        static const uint_t value = AlignmentBoundary ? Dimension+AlignmentBoundary-(Dimension%AlignmentBoundary) : Dimension;
    };

    template<ushort_t Boundary>
    struct aligned{
        static const ushort_t value=Boundary;
    };

    template<typename T>
    struct is_aligned : boost::mpl::false_{};

    template<ushort_t T>
    struct is_aligned<aligned<T> > : boost::mpl::true_ {};

}//namespace gridtools
