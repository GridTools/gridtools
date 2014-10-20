#pragma once
#include <boost/fusion/include/size.hpp>

namespace gridtools {

    namespace iterate_domain_aux {
        // struct assign_iterators {
        //     const uint_t i, j, k, bi, bj;

        //     template <typename IteratorType, typename StoragePointer>
        //     void assign (IteratorType & it, StoragePointer & storage) const {
        //         // std::cout << "Moving pointers **********************************************" << std::endl;
        //         // std::cout << typename std::remove_pointer<typename std::remove_const<typename std::remove_reference<StoragePointer>::type>::type>::type() << std::endl;
        //         // storage.info();
        //         it = &( storage(i,j,k) );
        //     }

        //     template <enumtype::backend Backend
        //               , typename IteratorType
        //               , typename ValueType
        //               , typename Layout
        //               , uint_t TileI
        //               , uint_t TileJ
        //               , uint_t MinusI
        //               , uint_t MinusJ
        //               , uint_t PlusI
        //               , uint_t PlusJ
        //               >
        //     void assign(IteratorType & it,
        //                 host_tmp_storage<
        //                 Backend
        //                 , ValueType
        //                 , Layout
        //                 , TileI
        //                 , TileJ
        //                 , MinusI
        //                 , MinusJ
        //                 , PlusI
        //                 , PlusJ
        //                 > & storage) const {
	    //    //std::cout << i << " - " << bi << " * " << TileI << std::endl;
	    //    //std::cout << j << " - " << bj << " * " << TileJ << std::endl;
	    //    it = storage.move_to(i - bi * TileI, j - bj * TileJ, k);
        //     }

        //     GT_FUNCTION
        //     assign_iterators(uint_t i, uint_t j, uint_t k, uint_t bi, uint_t bj)
        //         : i(i)
        //         , j(j)
        //         , k(k)
        //         , bi(bi)
        //         , bj(bj)
        //     {}

        //     template <typename ZipElem>
        //     GT_FUNCTION
        //     void operator()(ZipElem const & ze) const {
        //         // std::cout << "Moving pointers **********************************************" << std::endl;
        //         // std::cout << typename boost::remove_pointer<typename boost::remove_const<typename boost::remove_reference<typename boost::fusion::result_of::at_c<ZipElem, 1>::type>::type>::type>::type() << std::endl;
        //         // (*(boost::fusion::at_c<1>(ze))).info();

	    //   assign(boost::fusion::at_c<0>(ze),(*(boost::fusion::at_c<1>(ze))));
        //     }

        // };

        struct increment {
            template <typename Iterator>
            GT_FUNCTION
            void operator()(Iterator & it) const {
                it.value+=it.stride;
            }
        };

        struct decrement {
            template <typename Iterator>
            GT_FUNCTION
            void operator()(Iterator & it) const {
                it.value-=it.stride;
            }
        };

    } // namespace iterate_domain_aux

      namespace{
	template<uint_t ID>
	  struct assign_storage{
	  template<typename Left, typename Right>
	  GT_FUNCTION
	  static void inline assign(Left& l, Right & r, uint_t i, uint_t j, uint_t k){
	    boost::fusion::at_c<ID>(l).value=&((*boost::fusion::at_c<ID>(r))(i,j,k));
	    boost::fusion::at_c<ID>(l).stride=boost::fusion::at_c<ID>(r)->stride_k();
	    assign_storage<ID-1>::assign(l,r,i,j,k); //tail recursion
	    }
	};

	template<>
	  struct assign_storage<0>{
	  template<typename Left, typename Right>
	  GT_FUNCTION
	  static void inline assign(Left & l, Right & r, uint_t i, uint_t j, uint_t k){
          boost::fusion::at_c<0>(l).value=&((*boost::fusion::at_c<0>(r))(i,j,k));
          boost::fusion::at_c<0>(l).stride=boost::fusion::at_c<0>(r)->stride_k();
	  }
	};
      }

    template <typename LocalDomain>
    struct iterate_domain {
      typedef typename LocalDomain::local_iterators_type local_iterators_type;
      typedef typename LocalDomain::local_args_type local_args_type;
      static const uint_t N=boost::mpl::size<local_args_type>::value;
        LocalDomain const& local_domain;
        mutable local_iterators_type local_iterators;

        GT_FUNCTION
        iterate_domain(LocalDomain const& local_domain, uint_t i, uint_t j, uint_t k, uint_t bi, uint_t bj)
            : local_domain(local_domain)
        {                                 // double*            &storage
            m_i=i;
            m_j=j;
            m_k=k;
            //assign_storage< N-1 >::assign(local_iterators, local_domain.local_args, m_i, m_j, k);
            // DOUBLE*                                 &storage
	   /* boost::fusion::at_c<0>(local_iterators).value=&((*(boost::fusion::at_c<0>(local_domain.local_args)))(i,j,k)); */
	   /* boost::fusion::at_c<1>(local_iterators).value=&((*(boost::fusion::at_c<1>(local_domain.local_args)))(i,j,k)); */
	   /* boost::fusion::at_c<0>(local_iterators).stride=(*boost::fusion::at_c<0>(local_domain.local_args)).stride_k(); */
	   /* boost::fusion::at_c<1>(local_iterators).stride=(*boost::fusion::at_c<1>(local_domain.local_args)).stride_k(); */

	   /* printf("strides: %d\n", boost::fusion::at_c<0>(local_domain.local_args)->stride_k()); */
	   /* printf("strides: %d\n", boost::fusion::at_c<1>(local_domain.local_args)->stride_k()); */

        }


        GT_FUNCTION
        void increment() {
            //boost::fusion::for_each(local_iterators, iterate_domain_aux::increment());
            m_k++;
        }

        GT_FUNCTION
        void decrement() {
            //boost::fusion::for_each(local_iterators, iterate_domain_aux::decrement());
            m_k--;
        }

        template <typename T>
        GT_FUNCTION
        void info(T const &x) const {
            local_domain.info(x);
        }


        template <typename ArgType>
        typename boost::mpl::at<typename LocalDomain::esf_args, typename ArgType::index_type>::type::value_type& get_value(ArgType const& arg) const
            {
            // std::cout << " i " << arg.i()
            //           << " j " << arg.j()
            //           << " k " << arg.k()
            //           << " offset " << std::hex << (boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))->offset(arg.i(),arg.j(),arg.k()) << std::dec;
//                       << " base " << boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->min_addr()
//                       << " max_addr " << boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->max_addr()
//                       << " iterator " << boost::fusion::at<typename ArgType::index_type>(local_iterators)
//                       << " actual address " << boost::fusion::at<typename ArgType::index_type>(local_iterators)+(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))->offset(arg.i(),arg.j(),arg.k())
// //                      << " size of " << sizeof(typename boost::remove_pointer<typename boost::remove_reference<decltype(boost::fusion::at<typename ArgType::index_type>(local_iterators))>::type>::type)
//                 //<< " " << std::boolalpha << std::is_same<decltype(boost::fusion::at<typename ArgType::index_type>(local_iterators)), double*&>::type::value
//                       << " name " << boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->name()
//                       << std::endl;

            /* boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->info(); */


            assert(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->min_addr() <=
                   boost::fusion::at<typename ArgType::index_type>(local_iterators).value
                   +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                   ->offset(arg.i(),arg.j(),arg.k()));


            assert(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->max_addr() >
                   boost::fusion::at<typename ArgType::index_type>(local_iterators).value
                   +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                   ->offset(arg.i(),arg.j(),arg.k()));

                return *(boost::fusion::at<typename ArgType::index_type>(local_iterators).value/*(arg.template n<arg.n_args>())*/
                     +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                     ->offset(arg.i(),arg.j(),arg.k()));
            }



        template <typename ArgType>
        typename boost::mpl::at<typename LocalDomain::esf_args, typename ArgType::index_type>::type::value_type& get_value_(ArgType const& arg) const
            {
                auto storage_pointer= boost::fusion::at<typename ArgType::index_type>(local_domain.local_args);
                typedef typename std::remove_reference<decltype(*storage_pointer)>::type storage_type;
                // typename LocalDomain::iterator_type
                typename storage_type::iterator_type iterator;
                //set the storage iterator at the right position
                /*consdtexpr*/

                //printf("arg n function output (should be the offset) %d \n", arg.template n<arg.n_args>());
                iterator=&(storage_pointer->template get_address(arg.template n<arg.n_args>())[ storage_pointer->_index(m_i, m_j, m_k)]);
                //printf("i, j, k: %d, %d, %d\n", m_i, m_j, m_k);
                return *(iterator/*(arg.template n<arg.n_args>())*/
                     +storage_pointer->offset(arg.i(),arg.j(),arg.k()));
            }


/** @brief method called in the Do methods of the functors. */
        template <uint_t Index, typename Range>
        GT_FUNCTION
        typename boost::mpl::at<typename LocalDomain::esf_args, typename arg_type<Index, Range>::index_type>::type::value_type&
        operator()(arg_type<Index, Range> const& arg) const {
            // printf("normal \n\n\n\n");
            return get_value_(arg);
        }

/** @brief method called in the Do methods of the functors. */
        template <typename ArgType>
        GT_FUNCTION
        typename boost::mpl::at<typename LocalDomain::esf_args, typename ArgType::index_type>::type::value_type&
        operator()(gridtools::arg_decorator<ArgType> const& arg) const {
            // printf("integrator \n\n\n\n");
            return get_value_(arg);
        }


#ifdef CXX11_ENABLED
        template <typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto inline value(expr_plus<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) + (*this)(arg.second_operand)) {return (*this)(arg.first_operand) + (*this)(arg.second_operand);}

        template <typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto inline value(expr_minus<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) - (*this)(arg.second_operand)) {return (*this)(arg.first_operand) - (*this)(arg.second_operand);}

        template <typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto inline value(expr_times<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) * (*this)(arg.second_operand)) {return (*this)(arg.first_operand) * (*this)(arg.second_operand);}

        template <typename ArgType1, typename ArgType2>
        GT_FUNCTION
        auto inline value(expr_divide<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) / (*this)(arg.second_operand)) {return (*this)(arg.first_operand) / (*this)(arg.second_operand);}

        //partial specialisations for double (or float)
        template <typename ArgType1>
        GT_FUNCTION
        auto inline value(expr_plus<ArgType1, float_type> const& arg) const -> decltype((*this)(arg.first_operand) + arg.second_operand) {return (*this)(arg.first_operand) + arg.second_operand;}

        template <typename ArgType1>
        GT_FUNCTION
        auto inline value(expr_minus<ArgType1, float_type> const& arg) const -> decltype((*this)(arg.first_operand) - arg.second_operand) {return (*this)(arg.first_operand) - arg.second_operand;}

        template <typename ArgType1>
        GT_FUNCTION
        auto inline value(expr_times<ArgType1, float_type> const& arg) const -> decltype((*this)(arg.first_operand) * arg.second_operand) {return (*this)(arg.first_operand) * arg.second_operand;}

        template <typename ArgType1>
        GT_FUNCTION
        auto inline value(expr_divide<ArgType1, float_type> const& arg) const -> decltype((*this)(arg.first_operand) / arg.second_operand) {return (*this)(arg.first_operand) / arg.second_operand;}

/** @brief method called in the Do methods of the functors. */
        template <typename Expression >
        GT_FUNCTION
        auto operator() (Expression const& arg) const ->decltype(this->value(arg)) {
            return value(arg);
        }
#endif

    private:
        // iterate_domain remembers the state. This is necessary when we do finite differences and don't want to recompute all the iterators (but simply use the ones available for the current iteration storage for all the other storages)
        uint_t m_i;
        uint_t m_j;
        uint_t m_k;
    };

} // namespace gridtools
