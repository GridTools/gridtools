# Gridtools Storage Module

This repository contains the GridTools storage module. 

## Building:
```bash
# clone the repo
git clone https://github.com/eth-cscs/gridtools_storage gridtools_storage
# goto directory
cd gridtools_storage
# create and enter build directory
mkdir build
cd build
# call cmake 
cmake ..
# optionally call ccmake (e.g., activate GPU build)
ccmake .
# once done, call make to build everything
make
# execute unit tests
make test
```

## Main components:

* **Layout maps:** A layout map describes a memory layout. This can probably be illustrated with a simple example:
    * layout_map<0,1> is a layout with the first stride in the second dimension. If you have an array with this layout and you want to access the data the same order it is located in memory you would use 
    
    ```c++
    arr[0][0], arr[0][1], ..., arr[0][m], ..., arr[n][m]
    ```

    * layout_map<1,0> is a layout with the first stride in the first dimension. If you have an array with this layout and you want to access the data the same order it is located in memory you would use 
    
    ```c++
    arr[0][0], arr[1][0], ..., arr[n][0], ..., arr[n][m]
    ```

    The layout_map does not get instantiated and serves as a type that keeps static information only.

* **Storage Info:** A storage info is a class that is templated with an ID and a layout_map. So, lets say we want to describe something that stores double values in a layout with last stride on third dimension. This would be written as:

    ```c++
    const static unsigned Id = 0;
    typedef storage_info< Id, layout_map<0, 1, 2> > s_info_t;
    ```

    The storage_info object has to be instantiated. The constructor takes the size of the dimensions as arguments.
    ```c++
    s_info_t si(128, 128, 80);
    ```

    The reason why we are using an ID is to distinguish (by type) storage_infos with the same layout but different sizes.

* **Data store:** The data_store can be seen as the element that is coupling together the storage_info and a data type. As visible below the data_store is templated with a storage type and a storage_info type. The storage type can either be a host_storage or a cuda_storage (no worries, the selection is done by the storage-facility that is described below).

    ```c++
    // example of a host storage that holds double values 
    const static unsigned Id = 0;
    typedef storage_info< Id, layout_map<0, 1, 2> > s_info_t;
    s_info_t si(128, 128, 80);
    
    typedef data_store< host_storage<double>, sinfo_t > data_store_t;
    data_store_t ds(si);
    ```
    The data_store elements are holding a shared_ptr to the data and therefore can be copied without copying the data itself.
    ```c++
    // create a copy of the data_store
    data_store_t ds_cpy = ds;
    // all data-modifications will be visible to both ds and ds_cpy
    ```
    
* **Data store field:** A data store field is a collection of data_stores (of the same type). This can be convenient if somebody wants to store related fields (e.g., windspeed components). In order to keep this related fields in one element we introduced the data_store_field type. The definition looks like:
    ```c++
    // example of a host storage that holds double values 
    const static unsigned Id = 0;
    typedef storage_info< Id, layout_map<0, 1, 2> > s_info_t;
    s_info_t si(128, 128, 80);

    // create a field of data_store_t with 3 components (all of size 1)
    typedef data_store< host_storage<double>, sinfo_t > data_store_t;
    typedef data_store_field< data_store_t, 1, 1, 1 > data_store_field_t;
    data_store_field_t ds(si);
    ```
    The variadic numeric list that follows the data_store_t describes the size of the components. The number of values will be the number of components. So lets say we want to store 3 time steps of the windspeed components:
    ```c++
    ...
    // create a field of data_store_t with 3 components (all of size 3)
    typedef data_store< host_storage<double>, sinfo_t > data_store_t;
    typedef data_store_field< data_store_t, 3, 3, 3 > data_store_field_t;
    data_store_field_t ds(si);
    ```

* **Views:** The view provides means to modify the data stored in a data_store or data_store_field. Views can be generated (at the moment) for both Host (cpu) and Device (gpu). But it is very easy to add an extra "backend" that also supports other platforms (e.g., Xeon Phi). An example of how to generate a view:

    ```c++
    ...
    // instantiate a data_store and a data_store_field
    typedef data_store< host_storage<double>, sinfo_t > data_store_t;
    typedef data_store_field< data_store_t, 3, 3, 3 > data_store_field_t;

    data_store_t ds(si);
    data_store_field_t dsf(si);
    
    // create a view to ds (data_store)
    auto host_view_ds = make_host_view(ds);
    auto device_view_ds = make_device_view(ds);
    
    // set (0,10,10) to 3.1415
    host_view_ds(0, 10, 10) = 3.1415;
    
    // create a view to dsf (data_store_field)
    auto host_field_view_ds = make_host_field_view(ds);
    auto device_field_view_ds = make_device_field_view(ds);
    
    // set (0,10,10) of component (1,1) to 3.1415
    host_field_view_ds<1,1>(0, 10, 10) = 3.1415;
    ```
    Views can be set to readable. Which protects from unneeded data synchronization between the host and the device.
    In order to see how this works please have a look at the examples provided as unit tests.
    
## Code modifications:
Please consider following basic guidelines when extending/modifying the code.
* Provide a unit test that covers your changes!
* Use clang-format before committing any changes
* Don't push directly into master. Create a branch and pull request.
* Check compiler output and make sure that there are no new warnings and errors (check NVCC in particular). 
