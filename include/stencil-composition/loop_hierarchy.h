#include <iostream>
#include <common/defs.h>
/**@file this file implements the management of loops order

   The characteristics of the problems using this strategy are the following
   * arbitrary number of nested loops
   * a single kernel functor executed in the innermost loop
   * arbitrary direction of the loop (forward or backward)
   TODO: test this because probably it doesn't work yet

   The memory is addressed by a single index, which must be computed given all the loop indices.
   In order to reduce the computational complexity we want to increment such index in the nested
   loops instead of recomputing it each time. To this aim each loop item stores a copy of the index,
   which helps restoring the index to the previous state after arbitrary number of nesting.
*/

namespace gridtools{

    /**@class holding one loop

       It consists of the loop bounds, the step (whose default value is set to 1), the execution type (e.g. forward or backward), and an index identifying the space dimension this loop is acting on.
     */
    template<ushort_t ID, enumtype::execution Execution, uint_t Step=1>
    struct loop_item{

        /**@brief constructor*/
        GT_FUNCTION
        constexpr loop_item(uint_t const& low_bound, uint_t const& up_bound):
            m_up_bound(up_bound),
            m_low_bound(low_bound)
            {};

        /**@brief getter for the upper bound */
        GT_FUNCTION
        constexpr uint_t const&  up_bound() const {return m_up_bound; }

        /**@brief getter for the lower bound */
        GT_FUNCTION
        constexpr uint_t const&  low_bound() const {return m_low_bound; }

        /**@brief getter for the step */
        GT_FUNCTION
        constexpr uint_t step(){return s_step; }
        static const ushort_t s_id=ID;
        static const enumtype::execution s_execution=Execution;

    private:
        static const uint_t s_step=Step;
        loop_item();
        loop_item(loop_item const& other);
        uint_t m_up_bound;
        uint_t m_low_bound;
    };

    /**@class this class has the only purpose of comparing with the static case*/
    template<ushort_t ID, enumtype::execution Execution, uint_t LowBound, uint_t UpBound, uint_t Step=1>
struct static_loop_item{
    constexpr uint_t up_bound() const {return UpBound; }
    constexpr uint_t low_bound() const {return LowBound; }
    constexpr uint_t step(){return s_step; }
    static const uint_t s_step=Step;
    static const  ushort_t s_id=ID;
    static const enumtype::execution s_execution=Execution;
};

/**@class Main class for handling the loop hierarchy

It provides interfaces for initializing the loop and applying it.
*/
template<typename Array, typename First, typename ... Order>
struct loop_hierarchy : public loop_hierarchy<Array, Order ... > {
private:
    First loop;
    Array restore_index;
    typedef loop_hierarchy<Array, Order ...> next;

public:

    /**@brief constructor

       The constructor is templated with a pack of loop items, while it takes the loop items bounds as argument, so the first two arguments will correspond to the loop bounds (low and up respectively) of the first loop item. Note that if some of the loop items are static (static_loop_item) then their loop bounds are available from their types, and must not be specified as arguments.
     */
    template <typename ... ExtraArgs>
    GT_FUNCTION
    constexpr loop_hierarchy(ushort_t const& low_bound, ushort_t const& up_bound, ExtraArgs const& ... extra ) : next(extra...), loop(low_bound, up_bound)
        {
            //GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)>=sizeof...(Order), "not enough arguments passed to the constructor of the loops hierarchy")
            GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)/2<=sizeof...(Order), "too many arguments passed to the constructor of the loops hierarchy")
        }

    /**@brief constructor

       This constructor hopefully gets chosen when First is of type static_loop_item. Otherwise we have to enforce it with an enable_if.
     */
    template <typename ... ExtraArgs>
    GT_FUNCTION
    constexpr loop_hierarchy(ExtraArgs const& ... extra ) : next(extra...), loop()
        {
            //GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)>=sizeof...(Order), "not enough arguments passed to the constructor of the loops hierarchy")
            GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)<=sizeof...(Order), "too many arguments passed to the constructor of the loops hierarchy")
        }

    template <typename IterateDomain, typename BlockIdVector>
    GT_FUNCTION
    void
    initialize(IterateDomain & it_domain, BlockIdVector const& block_id){
#if defined(VERBOSE) && !defined(NDEBUG)
        std::cout<<"initialize the iteration space "<< First::s_id << " with " << loop.low_bound() << std::endl;
#endif
        it_domain.template initialize<First::s_id>(loop.low_bound(), block_id[First::s_id]);
        next::initialize(it_domain, block_id);
        it_domain.get_index(restore_index);
    }

    /**@brief running the loop

       This method executes the loop for the current level and calls recursively apply for
       the subsequent nested levels. Note that it also sets the iteration space in the variable
       "restore_index". This is necessary in order to restore the memory address index before
       incrementing it, and accounts for the fact that execution and strides can be in arbitrary order.
     */
    template <typename IterateDomain, typename InnerMostFunctor>
    GT_FUNCTION
    void apply( IterateDomain& it_domain, InnerMostFunctor const& kernel){
        for (uint_t i=loop.low_bound(); i<=loop.up_bound(); i+=loop.step())
        {
#if defined(VERBOSE) && !defined(NDEBUG)
            std::cout<<"iteration "<<i<<", index "<<First::s_id<<std::endl;
#endif
            next::apply(it_domain, kernel);
            it_domain.set_index(restore_index);//redundant in the last iteration
            it_domain.template increment<First::s_id, First::s_execution>();//redundant in the last iteration
            update_index(it_domain);
        }
    }

    /**@brief updating the restore_index with the current value*/
    template <typename IterateDomain>
    GT_FUNCTION
    void update_index( IterateDomain const& it_domain ){
#if defined(VERBOSE) && !defined(NDEBUG)
        std::cout<<"updating the index for level "<<First::s_id<<std::endl;
#endif
        it_domain.get_index(restore_index);//redundant in the last iteration
        next::update_index(it_domain);//redundant in the last iteration
    }
};

    /**@class Specialization for the case of the innermost loop

       TODO: there is some code repetition here below
     */
template<typename Array, typename First>
struct loop_hierarchy<Array, First> {

private:
    First loop;
    Array restore_index;

public:
    GT_FUNCTION
    constexpr loop_hierarchy(ushort_t const& up_bound, ushort_t const& low_bound ): loop(up_bound, low_bound)
        {
        }

    GT_FUNCTION
    constexpr loop_hierarchy( ) : loop()
        {
        }

    /**@brief executes the loop

       This represents the innermost loop, thus it executes the funcctor which is passed on by
       all previous levels.
     */
    template <typename IterateDomain, typename InnerMostFunctor>
    GT_FUNCTION
    void apply(IterateDomain & it_domain, InnerMostFunctor const& kernel){
        for (uint_t i=loop.low_bound(); i<=loop.up_bound(); i+=loop.step())
        {
#if defined(VERBOSE) && !defined(NDEBUG)
            std::cout<<"iteration "<<i<<", last index "<<First::s_id<<std::endl;
#endif
            kernel();
            it_domain.set_index(restore_index);//redundant in the last iteration
            it_domain.template increment<First::s_id, First::s_execution>();
            update_index(it_domain);
        }
    }

    /**@brief updating the restore_index with the current value*/
    template <typename IterateDomain>
    GT_FUNCTION
    void update_index( IterateDomain const& it_domain ){
#if defined(VERBOSE) && !defined(NDEBUG)
        std::cout<<"updating the index for level "<<First::s_id<<std::endl;
#endif

        it_domain.get_index(restore_index);//redundant in the last iteration
    }

    template <typename IterateDomain, typename BlockIdVector>
    GT_FUNCTION
    void
    initialize(IterateDomain & it_domain, BlockIdVector const& block_id){

#if defined(VERBOSE) && !defined(NDEBUG)
        std::cout<<"initialize the iteration space "<< First::s_id << " with " << loop.low_bound() << std::endl;
#endif
        it_domain.template initialize<First::s_id>(loop.low_bound(), block_id[First::s_id]);
        it_domain.get_index(restore_index);
    }
};















//     template<typename Array, typename First, typename ... Order>
//     struct loop_hierarchy_naive
//     {
//     private:
//         First loop;
//         typedef loop_hierarchy<Array, Order ...> next;

//     public:

//         /**@brief constructor

//            The constructor is templated with a pack of loop items, while it takes the loop items bounds as argument, so the first two arguments will correspond to the loop bounds (low and up respectively) of the first loop item. Note that if some of the loop items are static (static_loop_item) then their loop bounds are available from their types, and must not be specified as arguments.
//      */
//         template <typename ... ExtraArgs>
//         GT_FUNCTION
//         constexpr loop_hierarchy_naive(ushort_t const& low_bound, ushort_t const& up_bound, ExtraArgs const& ... extra ) : next(extra...), loop(low_bound, up_bound)
//         {
//             //GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)>=sizeof...(Order), "not enough arguments passed to the constructor of the loops hierarchy")
//             GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)/2<=sizeof...(Order), "too many arguments passed to the constructor of the loops hierarchy")
//                 }

//         /**@brief constructor

//            This constructor hopefully gets chosen when First is of type static_loop_item. Otherwise we have to enforce it with an enable_if.
//         */
//         template <typename ... ExtraArgs>
//     GT_FUNCTION
//         constexpr loop_hierarchy_naive(ExtraArgs const& ... extra ) : next(extra...), loop()
//             {
//                 //GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)>=sizeof...(Order), "not enough arguments passed to the constructor of the loops hierarchy")
//             GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)<=sizeof...(Order), "too many arguments passed to the constructor of the loops hierarchy")
//         }

//     template <typename IterateDomain>
//     GT_FUNCTION
//     void
//     initialize(IterateDomain & it_domain){
// #if defined(VERBOSE) && !defined(NDEBUG)
//         std::cout<<"initialize the iteration space "<< First::s_id << " with " << loop.low_bound() << std::endl;
// #endif
//         it_domain.template increment<First::s_id, First::s_execution>(loop.low_bound()+kernel.functor()->m_block_id[First::s_id]);
//         next::initialize(it_domain);
//     }

//     /**@brief running the loop

//        This method executes the loop for the current level and calls recursively apply for
//        the subsequent nested levels. Note that it also sets the iteration space in the variable
//        "restore_index". This is necessary in order to restore the memory address index before
//        incrementing it, and accounts for the fact that execution and strides can be in arbitrary order.
//      */
//     template <typename IterateDomain, typename InnerMostFunctor>
//     GT_FUNCTION
//     void apply( IterateDomain& it_domain, InnerMostFunctor const& kernel){
//         for (uint_t i=loop.low_bound(); i<=loop.up_bound(); i+=loop.step())
//         {
// #if defined(VERBOSE) && !defined(NDEBUG)
//             std::cout<<"iteration "<<i<<", index "<<First::s_id<<std::endl;
// #endif
//             next::apply(it_domain, kernel);
//         }
//     }

//     /**@brief updating the restore_index with the current value*/
//     template <typename IterateDomain>
//     GT_FUNCTION
//     void update_index( IterateDomain const& it_domain ){
// #if defined(VERBOSE) && !defined(NDEBUG)
//         std::cout<<"updating the index for level "<<First::s_id<<std::endl;
// #endif
//     }

//     };


//     /**@class Specialization for the case of the innermost loop

//        TODO: there is some code repetition here below
//     */
//     template<typename Array, typename First>
//     struct loop_hierarchy<Array, First> {

//     private:
//         First loop;

//     public:
//         GT_FUNCTION
//         constexpr loop_hierarchy_naive(ushort_t const& up_bound, ushort_t const& low_bound ): loop(up_bound, low_bound)
//         {
//         }

//         GT_FUNCTION
//         constexpr loop_hierarchy_naive( ) : loop()
//         {
//         }

//         /**@brief executes the loop

//            This represents the innermost loop, thus it executes the funcctor which is passed on by
//            all previous levels.
//         */
//         template <typename IterateDomain, typename InnerMostFunctor>
//         GT_FUNCTION
//         void apply(IterateDomain & it_domain, InnerMostFunctor const& kernel, uint_t const& i ){
//             for (uint_t j=loop.low_bound(); j<=loop.up_bound(); j+=loop.step())
//             {
// #if defined(VERBOSE) && !defined(NDEBUG)
//                 std::cout<<"iteration "<<i<<", last index "<<First::s_id<<std::endl;
// #endif
//                 kernel();
//                 it_domain.template increment<0, enumtype::forward>(i, kernel.functor()->m_block_id[ID]);
//                 it_domain.template increment<1, enumtype::forward>(j, kernel.functor()->m_block_id[ID]);
//         }
//         }

//         /**@brief updating the restore_index with the current value*/
//     template <typename IterateDomain>
//     GT_FUNCTION
//     void update_index( IterateDomain const& it_domain ){
// #if defined(VERBOSE) && !defined(NDEBUG)
//         std::cout<<"updating the index for level "<<First::s_id<<std::endl;
// #endif

//         it_domain.get_index(restore_index);//redundant in the last iteration
//     }

//         template <typename IterateDomain>
//         GT_FUNCTION
//         void
//         initialize(IterateDomain & it_domain){

// #if defined(VERBOSE) && !defined(NDEBUG)
//             std::cout<<"initialize the iteration space "<< First::s_id << " with " << loop.low_bound() << std::endl;
// #endif
//             it_domain.template increment<First::s_id, First::s_execution>(loop.low_bound()+kernel.functor()->m_block_id[First::s_id]);
//             it_domain.get_index(restore_index);
//         }
// };

//     template<ushort_t ID>
//     struct increment_all{

//         template<typename IterateDomain, typename InnermostFunctor>
//         void apply(IterateDomain const& it_domain, InnermostFunctor const& kernel, array<uint_t, IterateDomain::>  steps_){
//             it_domain.template increment<ID, enumtype::forward>(steps_, kernel.functor()->m_block_id[ID]);
//             increment_all<ID-1>::apply(it_domain, );
//         }
//     };

//         template<>
//         struct increment_all<0>{

//         template<typename IterateDomain, typename InnermostFunctor>
//         void apply(IterateDomain const& it_domain, InnermostFunctor const& kernel, uint_t const& steps_){
//             it_domain.template increment<0, enumtype::forward>(steps_, kernel.functor()->m_block_id[0]);
//         }
//     };


}//namespace gridtools
