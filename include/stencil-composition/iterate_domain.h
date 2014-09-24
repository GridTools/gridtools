#pragma once

#include <boost/fusion/view/zip_view.hpp>

namespace gridtools {

    namespace iterate_domain_aux {
        struct assign_iterators {
            const int i, j, k, bi, bj;

            template <typename IteratorType, typename StoragePointer>
            void assign (IteratorType & it, StoragePointer & storage) const {
                // std::cout << "Moving pointers **********************************************" << std::endl;
                // std::cout << typename std::remove_pointer<typename std::remove_const<typename std::remove_reference<StoragePointer>::type>::type>::type() << std::endl;
                // storage.info();
                it = &( storage(i,j,k) );
            }

            template <enumtype::backend Backend
                      , typename IteratorType
                      , typename ValueType
                      , typename Layout
                      , int TileI
                      , int TileJ
                      , int MinusI
                      , int MinusJ
                      , int PlusI
                      , int PlusJ
                      >
            void assign(IteratorType & it,
                        host_tmp_storage<
                        Backend
                        , ValueType
                        , Layout
                        , TileI
                        , TileJ
                        , MinusI
                        , MinusJ
                        , PlusI
                        , PlusJ
                        > & storage) const {
                // std::cout << i << " - " << bi << " * " << TileI << std::endl;
                // std::cout << j << " - " << bj << " * " << TileJ << std::endl;
                it = storage.move_to(i - bi * TileI, j - bj * TileJ, k);
            }

            GT_FUNCTION
            assign_iterators(int i, int j, int k, int bi, int bj)
                : i(i)
                , j(j)
                , k(k)
                , bi(bi)
                , bj(bj)
            {}

            template <typename ZipElem>
            GT_FUNCTION
            void operator()(ZipElem const & ze) const {
                // std::cout << "Moving pointers **********************************************" << std::endl;
                // std::cout << typename boost::remove_pointer<typename boost::remove_const<typename boost::remove_reference<typename boost::fusion::result_of::at_c<ZipElem, 1>::type>::type>::type>::type() << std::endl;
                // (*(boost::fusion::at_c<1>(ze))).info();

                assign(boost::fusion::at_c<0>(ze),(*(boost::fusion::at_c<1>(ze))));
            }

        };

        struct increment {
            template <typename Iterator>
            GT_FUNCTION
            void operator()(Iterator & it) const {
                ++it;
            }
        };

        struct decrement {
            template <typename Iterator>
            GT_FUNCTION
            void operator()(Iterator & it) const {
                --it;
            }
        };

    } // namespace iterate_domain_aux

    template <typename LocalDomain>
    struct iterate_domain {
        typedef typename LocalDomain::local_iterators_type local_iterators_type;

        LocalDomain const& local_domain;
        mutable local_iterators_type local_iterators;

        GT_FUNCTION
        iterate_domain(LocalDomain const& local_domain, int i, int j, int k, int bi, int bj)
            : local_domain(local_domain)
        {
            typedef boost::fusion::vector<local_iterators_type&, typename LocalDomain::local_args_type const &> to_zip;
            typedef boost::fusion::zip_view<to_zip> zipping;

            to_zip z(local_iterators, local_domain.local_args);
            zipping zipped(z);
            boost::fusion::for_each(zipped, iterate_domain_aux::assign_iterators(i,j,k,bi,bj));
        }

        GT_FUNCTION
        void increment() const {
            boost::fusion::for_each(local_iterators, iterate_domain_aux::increment());
        }

        GT_FUNCTION
        void decrement() const {
            boost::fusion::for_each(local_iterators, iterate_domain_aux::decrement());
        }

        template <typename T>
        GT_FUNCTION
        void info(T const &x) const {
            local_domain.info(x);
        }

/** @brief method called in the Do methods of the functors. */
        template <int Index, typename Range>
        GT_FUNCTION
        typename boost::mpl::at<typename LocalDomain::esf_args, typename arg_type<Index, Range>::index_type>::type::value_type&
        operator()(arg_type<Index, Range> const& arg) const {
            typedef arg_type<Index, Range> ArgType;

//             std::cout << " i " << arg.i()
//                       << " j " << arg.j()
//                       << " k " << arg.k()
//                       << " offset " << std::hex << (boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))->offset(arg.i(),arg.j(),arg.k()) << std::dec
//                       << " base " << boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->min_addr()
//                       << " max_addr " << boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->max_addr()
//                       << " iterator " << boost::fusion::at<typename ArgType::index_type>(local_iterators)
//                       << " actual address " << boost::fusion::at<typename ArgType::index_type>(local_iterators)+(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))->offset(arg.i(),arg.j(),arg.k())
// //                      << " size of " << sizeof(typename boost::remove_pointer<typename boost::remove_reference<decltype(boost::fusion::at<typename ArgType::index_type>(local_iterators))>::type>::type)
//                 //<< " " << std::boolalpha << std::is_same<decltype(boost::fusion::at<typename ArgType::index_type>(local_iterators)), double*&>::type::value
//                       << " name " << boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->name()
//                       << std::endl;

            boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->info();

            assert(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->min_addr() <=
                   boost::fusion::at<typename ArgType::index_type>(local_iterators)
                   +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                   ->offset(arg.i(),arg.j(),arg.k()));

            assert(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args)->max_addr() >
                   boost::fusion::at<typename ArgType::index_type>(local_iterators)
                   +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                   ->offset(arg.i(),arg.j(),arg.k()));

            return *(boost::fusion::at<typename ArgType::index_type>(local_iterators)
                     +(boost::fusion::at<typename ArgType::index_type>(local_domain.local_args))
                     ->offset(arg.i(),arg.j(),arg.k()));
        }




        template <typename ArgType1, typename ArgType2>
        auto inline value(expr_plus<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) + (*this)(arg.second_operand)) {return (*this)(arg.first_operand) + (*this)(arg.second_operand);}

        template <typename ArgType1, typename ArgType2>
        auto inline value(expr_minus<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) - (*this)(arg.second_operand)) {return (*this)(arg.first_operand) - (*this)(arg.second_operand);}

        template <typename ArgType1, typename ArgType2>
        auto inline value(expr_times<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) * (*this)(arg.second_operand)) {return (*this)(arg.first_operand) * (*this)(arg.second_operand);}

        template <typename ArgType1, typename ArgType2>
        auto inline value(expr_divide<ArgType1, ArgType2> const& arg) const -> decltype((*this)(arg.first_operand) / (*this)(arg.second_operand)) {return (*this)(arg.first_operand) / (*this)(arg.second_operand);}


        // template <typename whatever>
        // auto value(whatever arg) const -> decltype((*this)(arg.first_operand)) const {return (*this)(arg.first_operand);}

        // double const value(const expr<gridtools::expr_times<gridtools::arg_type<3, gridtools::range<0, 0, 0, 0> >, gridtools::arg_type<1, gridtools::range<0, 0, 0, 0> > > > whatever) const {return 2.;}

/** @brief method called in the Do methods of the functors. */
        template <typename Expression >
        GT_FUNCTION
        auto operator() (Expression const& arg) const ->decltype(value(arg)) const {
            return value(arg);
        }

        // template <typename Expression >
        // GT_FUNCTION
        // double operator() (Expression const& arg)  const {
        //     return value(arg);
        // }


        // template <typename whatever>
        // double const value(whatever const arg) const {return 0.;}


    };

} // namespace gridtools
