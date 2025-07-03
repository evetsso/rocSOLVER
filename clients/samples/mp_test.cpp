#include <H5Cpp.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "gpubuf.h"
#include "hip_object_wrapper.h"
#include "rocsolver/rocsolver.h"

int main(int argc, char** argv)
{
    size_t ntrial = std::stoull(argv[3]);

    // define double complex type
    H5::CompType ztype(sizeof(double) * 2);
    ztype.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
    ztype.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);

    // load files
    H5::H5File kkr(argv[1], H5F_ACC_RDONLY);
    H5::H5File t(argv[2], H5F_ACC_RDONLY);

    auto kkrmat = kkr.openDataSet("kkrmat");
    auto tmat = t.openDataSet("tmat");

    printf("kkr size = %zu\nt size = %zu\n", kkrmat.getInMemDataSize(), tmat.getInMemDataSize());

    auto kkrmat_type = kkrmat.getTypeClass();
    printf("kkr type is %d\n", kkrmat_type);
    auto tmat_type = tmat.getTypeClass();
    printf("t type is %d\n", tmat_type);

    auto kkrmat_space = kkrmat.getSpace();
    kkrmat_space.selectAll();
    auto tmat_space = tmat.getSpace();
    tmat_space.selectAll();

    std::vector<hsize_t> kkrmat_dims(kkrmat_space.getSimpleExtentNdims());
    std::vector<hsize_t> tmat_dims(tmat_space.getSimpleExtentNdims());
    kkrmat_space.getSimpleExtentDims(kkrmat_dims.data());
    tmat_space.getSimpleExtentDims(tmat_dims.data());

    printf("kkr dims = ");
    for(auto d : kkrmat_dims)
        printf("%llu ", d);
    printf("\nt dims = ");
    for(auto d : tmat_dims)
        printf("%llu ", d);
    printf("\n");

    // expecting two squares
    if(kkrmat_dims.size() != 2 || tmat_dims.size() != 2 || kkrmat_dims[0] != kkrmat_dims[1]
       || tmat_dims[0] != tmat_dims[1])
    {
        throw std::runtime_error("expected two squares");
    }

    // copy data to host memory
    auto kkrmat_data_host = std::make_unique<rocblas_double_complex[]>(
        kkrmat.getInMemDataSize() / sizeof(rocblas_double_complex));
    auto tmat_data_host = std::make_unique<rocblas_double_complex[]>(
        tmat.getInMemDataSize() / sizeof(rocblas_double_complex));

    kkrmat.read(kkrmat_data_host.get(), ztype);
    printf("read kkr successfully\n");
    tmat.read(tmat_data_host.get(), ztype);
    printf("read tmat successfully\n");

    // padded tmat out to have same number of columns as kkr
    auto tmat_data_pad_host = std::make_unique<rocblas_double_complex[]>(
        kkrmat_dims[0] * tmat_dims[0] * sizeof(rocblas_double_complex));
    std::fill_n(tmat_data_pad_host.get(), kkrmat_dims[0] * tmat_dims[0],
                rocblas_double_complex{0.0, 0.0});
    // copy tmat to first part of padded buffer
    for(hsize_t trow = 0; trow < tmat_dims[0]; ++trow)
    {
        std::copy_n(tmat_data_host.get() + trow * tmat_dims[1], tmat_dims[0],
                    tmat_data_pad_host.get() + trow * kkrmat_dims[1]);
    }

    gpubuf_t<rocblas_double_complex> kkrmat_data_device;
    gpubuf_t<rocblas_double_complex> tmat_data_device;
    if(kkrmat_data_device.alloc(kkrmat.getInMemDataSize()) != hipSuccess)
        throw std::runtime_error("failed to hipmalloc kkr");
    if(tmat_data_device.alloc(kkrmat_dims[0] * tmat_dims[0] * sizeof(rocblas_double_complex))
       != hipSuccess)
        throw std::runtime_error("failed to hipmalloc t");

    gpubuf_t<rocblas_int> ipiv;
    if(ipiv.alloc(sizeof(rocblas_int) * kkrmat_dims[0]) != hipSuccess)
        throw std::runtime_error("failed to hipmalloc ipiv");

    gpubuf_t<rocblas_int> info;
    if(info.alloc(sizeof(rocblas_int)) != hipSuccess)
        throw std::runtime_error("failed to hipmalloc info");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    hipEvent_wrapper_t start, stop;
    start.alloc();
    stop.alloc();
    std::vector<float> gpu_time(ntrial);
    for(size_t i = 0; i < ntrial; ++i)
    {
        // copy to device
        if(hipMemcpy(kkrmat_data_device.data(), kkrmat_data_host.get(), kkrmat.getInMemDataSize(),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to memcpy kkr to device");
        if(hipMemcpy(tmat_data_device.data(), tmat_data_pad_host.get(),
                     kkrmat_dims[0] * tmat_dims[0] * sizeof(rocblas_double_complex),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to memcpy t to device");

        if(hipMemset(ipiv.data(), -1, ipiv.size()) != hipSuccess)
            throw std::runtime_error("failed to init ipiv");

        if(hipMemset(info.data(), -1, info.size()) != hipSuccess)
            throw std::runtime_error("failed to init info");

        (void)hipDeviceSynchronize();

        if(hipEventRecord(start) != hipSuccess)
            throw std::runtime_error("failed to record start event");

        // solve
        auto status = rocsolver_zgesv(handle, kkrmat_dims[0], tmat_dims[0],
                                      kkrmat_data_device.data(), kkrmat_dims[1], ipiv.data(),
                                      tmat_data_device.data(), kkrmat_dims[1], info.data());

        if(hipEventRecord(stop) != hipSuccess)
            throw std::runtime_error("failed to record stop event");
        if(hipEventSynchronize(stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");

        float time;
        if(hipEventElapsedTime(&time, start, stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failed");
        gpu_time[i] = time;

        rocblas_int info_host;
        if(hipMemcpy(&info_host, info.data(), sizeof(rocblas_int), hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("failed to copy info back");

        printf("trial %zu info=%d status=%d\n", i, info_host, status);
    }

    rocblas_destroy_handle(handle);
    handle = nullptr;

    printf("Execution times (ms):");
    for(auto t : gpu_time)
    {
        printf(" %.2f", static_cast<double>(t));
    }

    std::sort(gpu_time.begin(), gpu_time.end());
    if(ntrial % 2)
    {
        printf("\nMedian time (ms): %.2f\n", gpu_time[ntrial / 2]);
    }
    else
    {
        printf("\nMedian time (ms): %.2f\n", (gpu_time[ntrial / 2] + gpu_time[ntrial / 2 + 1]) / 2);
    }

    return 0;
}
