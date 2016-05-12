#pragma once
#include <iostream>
#include "common/defs.hpp"

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
    template<ushort_t ID, typename Integer=int_t, uint_t Step=1>
    struct loop_item{

        typedef Integer value_type;

        /**@brief constructor*/
        GT_FUNCTION
        constexpr loop_item(Integer const& low_bound, Integer const& up_bound):
            m_up_bound(up_bound),
            m_low_bound(low_bound)
            {};

        /**@brief getter for the upper bound */
        GT_FUNCTION
        constexpr Integer const&  up_bound() const {return m_up_bound; }

        /**@brief getter for the lower bound */
        GT_FUNCTION
        constexpr Integer const&  low_bound() const {return m_low_bound; }

        /**@brief getter for the step */
        GT_FUNCTION
        constexpr uint_t step(){return s_step; }
        static const ushort_t s_id=ID;

        static const uint_t s_step=Step;
    private:
        loop_item();
        loop_item(loop_item const& other);
        Integer m_up_bound;
        Integer m_low_bound;
    };

    /**@class this class is useful to compare with the case in which the loop
       bounds are static const*/
    template<ushort_t ID, uint_t LowBound, uint_t UpBound, typename Integer=int_t, uint_t Step=1>
    struct static_loop_item{
        typedef Integer value_type;
        constexpr Integer up_bound() const {return UpBound; }
        constexpr Integer low_bound() const {return LowBound; }
        constexpr uint_t step(){return s_step; }
        static const uint_t s_step=Step;
        static const  ushort_t s_id=ID;
    };

    template<typename Loop>
    struct is_static_loop;

    template<ushort_t ID, uint_t LowBound, uint_t UpBound, typename Integer, uint_t Step>
    struct is_static_loop< static_loop_item<ID// , Execution
                                            , LowBound, UpBound, Integer, Step> > : public boost::true_type
    {};

    template<ushort_t ID, typename Integer, uint_t Step>
    struct is_static_loop< loop_item<ID,Integer,  Step> > : public boost::false_type
    {};

    template<ushort_t ID, typename Integer>
    struct is_static_loop< loop_item<ID, Integer> > : public boost::false_type
    {};

    template<ushort_t ID>
    struct is_static_loop< loop_item<ID> > : public boost::false_type
    {};

#ifdef CXX11_ENABLED
/**@class Main class for handling the loop hierarchy

   It provides interfaces for initializing the loop and applying it.
   TODO: store the iterate_domain as a const reference
*/
    template<typename Array, typename First, typename ... Order>
    struct loop_hierarchy : public loop_hierarchy<Array, Order ... > {
    private:
        First loop;
        Array restore_index;
        typedef loop_hierarchy<Array, Order ...> next;
#else

        template <typename Array, typename First>
        struct loop_hierarchy0;
// only 2 nested loops possible when CXX11 is disabled
        /**@class Main class for handling the loop hierarchy

           It provides interfaces for initializing the loop and applying it.
        */
        template<typename Array, typename First, typename Order>
        struct loop_hierarchy : public loop_hierarchy0<Array, Order > {
        private:
            First loop;
            Array restore_index;
            typedef loop_hierarchy0<Array, Order> next;
#endif

    public:

            typedef typename First::value_type value_type;
#ifdef CXX11_ENABLED
        /**@brief constructor

           The constructor is templated with a pack of loop items, while it takes the loop items bounds as argument, so the first two arguments will correspond to the loop bounds (low and up respectively) of the first loop item. Note that if some of the loop items are static (static_loop_item) then their loop bounds are available from their types, and must not be specified as arguments.
        */
        template < typename ... ExtraArgs
                   , typename = std::enable_if< !(is_static_loop<First>::value) >
                   >
        GT_FUNCTION
        constexpr loop_hierarchy(value_type const& low_bound, value_type const& up_bound, ExtraArgs const& ... extra) : next(extra...), loop(low_bound, up_bound)
        {
            //GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)>=sizeof...(Order), "not enough arguments passed to the constructor of the loops hierarchy");
            GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)/2<=sizeof...(Order), "too many arguments passed to the constructor of the loops hierarchy");
        }

        /**@brief constructor

           This constructor hopefully gets chosen when First is of type static_loop_item. Otherwise we have to enforce it with an enable_if.
        */
        template < typename ... ExtraArgs
                   , typename = std::enable_if< is_static_loop<First>::value >
                   >
        GT_FUNCTION
        constexpr loop_hierarchy(ExtraArgs const& ... extra  ) : next(extra...), loop()
        {
            GRIDTOOLS_STATIC_ASSERT(sizeof...(ExtraArgs)<=sizeof...(Order), "too many arguments passed to the constructor of the loops hierarchy");
        }

#else //for CXX11_ENABLED==false only 2 nested loops are allowed

        GT_FUNCTION
        constexpr loop_hierarchy(value_type const& low_bound, value_type const& up_bound, value_type const& low_bound2, value_type const& up_bound2 ) : next(low_bound2, up_bound2), loop(low_bound, up_bound){}

        //for CXX11_ENABLED==false the static loops MUST be the last ones
        GT_FUNCTION
        constexpr loop_hierarchy(value_type const& low_bound, value_type const& up_bound)
            : next(), loop(low_bound, up_bound){}


#endif //CXX11_ENABLED

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
           NOTE: the loops are inclusive (including the two boundary values)
        */
        template <typename IterateDomain, typename InnerMostFunctor>
        GT_FUNCTION
        void apply( IterateDomain& it_domain, InnerMostFunctor & kernel){
            for (value_type i=loop.low_bound(); i<=loop.up_bound(); i+=loop.step())
            {
#if defined(VERBOSE) && !defined(NDEBUG)
                std::cout<<"iteration "<<i<<", index "<<First::s_id<<std::endl;
#endif
                next::apply(it_domain, kernel);
                it_domain.set_index(restore_index);//redundant in the last iteration
                it_domain.template increment<First::s_id, static_uint<First::s_step> >();//redundant in the last iteration
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
        private:
            loop_hierarchy();
            loop_hierarchy( const loop_hierarchy& other);
    };

    /**@class Specialization for the case of the innermost loop

       TODO: there is some code repetition here below
    */
    template<typename Array, typename First>
#ifdef CXX11_ENABLED
    struct loop_hierarchy<Array, First>
#else
    struct loop_hierarchy0
#endif
    {

    private:
        First loop;
        Array restore_index;

    public:
        typedef typename First::value_type value_type;
        GT_FUNCTION
        constexpr
#ifdef CXX11_ENABLED
        loop_hierarchy
#else
        loop_hierarchy0
#endif
(value_type const& up_bound, value_type const& low_bound ): loop(up_bound, low_bound)
            {
            }

        GT_FUNCTION
        constexpr
#ifdef CXX11_ENABLED
        loop_hierarchy
#else
        loop_hierarchy0
#endif
( ) : loop(){}

        /**@brief executes the loop

           This represents the innermost loop, thus it executes the funcctor which is passed on by
           all previous levels.
        */
        template <typename IterateDomain, typename InnerMostFunctor>
        GT_FUNCTION
        void apply(IterateDomain & it_domain, InnerMostFunctor & kernel){
            for (value_type i=loop.low_bound(); i<=loop.up_bound(); i+=loop.step())
            {
#if defined(VERBOSE) && !defined(NDEBUG)
                std::cout<<"iteration "<<i<<", index (last) "<<First::s_id<<std::endl;
#endif
                kernel();
                it_domain.set_index(restore_index);//redundant in the last iteration
                it_domain.template increment<First::s_id, static_uint<First::s_step> >();
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

}//namespace gridtools
