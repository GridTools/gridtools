#include "gtest/gtest.h"
#include <iostream>
#include <string>

#include <stencil-composition/stencil-composition.hpp>

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

TEST(storage, naming) {
    typedef gridtools::layout_map<0,1,2> layout_t;
    typedef backend<Host, GRIDBACKEND, Naive >::storage_info<__COUNTER__, layout_t> meta_data_t;
    typedef backend<Host, GRIDBACKEND, Naive >::storage_type<float_type, meta_data_t >::type storage_t;

    meta_data_t meta_data_(1,1,1);

    typedef storage_t storage_type;
    //create storage with name in
    storage_type in(meta_data_, "bla");
    EXPECT_EQ(std::string(in.get_name()), "bla");
    //set name to modified using const char *
    in.set_name("modified_ccp");
    EXPECT_EQ(std::string(in.get_name()), "modified_ccp");
    //set name using string that gets deleted afterwards
	{
		std::string t("strtmp");
		in.set_name(t.c_str());
		// t gets destroyed here
	}
	{
		std::string t("tmpstr"); // value of in.m_name maybe gets overridden here by accident
		//validate it is still the old string
		EXPECT_EQ(std::string(in.get_name()), "strtmp");
	}
}

