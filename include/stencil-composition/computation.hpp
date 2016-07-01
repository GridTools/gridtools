#pragma once
// //\todo this struct becomes redundant when the auto keyword is used
namespace gridtools {
    template < typename ReductionType = int >
    struct computation {
        virtual void ready() = 0;
        virtual void steady() = 0;
        virtual void finalize() = 0;
        virtual ReductionType run() = 0;
        virtual std::string print_meter() = 0;
        virtual double get_meter() = 0;
        virtual void reset_meter() = 0;
    };

} // namespace gridtools
