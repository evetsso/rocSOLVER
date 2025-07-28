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
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

#include <dlfcn.h>
#include <link.h>

struct VectorNorms
{
    double l_2 = 0.0, l_inf = 0.0;
};

int main(int argc, char** argv)
{
    std::string ref_lib_path = argv[1];
    std::string dev_lib_path = argv[2];

    auto ref_lib = dlopen(ref_lib_path.c_str(), RTLD_LAZY);
    auto dev_lib = dlopen(dev_lib_path.c_str(), RTLD_LAZY);

    auto ref_zgesv = reinterpret_cast<decltype(&rocsolver_zgesv)>(dlsym(ref_lib, "rocsolver_zgesv"));
    auto dev_zgesv = reinterpret_cast<decltype(&rocsolver_zgesv)>(dlsym(dev_lib, "rocsolver_zgesv"));

    size_t ntrial = std::stoull(argv[5]);

    // define double complex type
    H5::CompType ztype(sizeof(double) * 2);
    ztype.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
    ztype.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);

    // load files
    H5::H5File kkr(argv[3], H5F_ACC_RDONLY);
    H5::H5File t(argv[4], H5F_ACC_RDONLY);

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
    std::vector<float> ref_gpu_time(ntrial);
    std::vector<float> dev_gpu_time(ntrial);

    std::unique_ptr<rocblas_double_complex[]> ref_output;
    std::unique_ptr<rocblas_double_complex[]> dev_output;

    for(size_t i = 0; i < ntrial; ++i)
    {
        for(auto run_ref : {true, false})
        {
            auto zgesv = run_ref ? ref_zgesv : dev_zgesv;
            auto& gpu_time = run_ref ? ref_gpu_time : dev_gpu_time;
            auto& output = run_ref ? ref_output : dev_output;

            // copy to device
            if(hipMemcpy(kkrmat_data_device.data(), kkrmat_data_host.get(),
                         kkrmat.getInMemDataSize(), hipMemcpyHostToDevice)
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
            auto status = zgesv(handle, kkrmat_dims[0], tmat_dims[0], kkrmat_data_device.data(),
                                kkrmat_dims[1], ipiv.data(), tmat_data_device.data(),
                                kkrmat_dims[1], info.data());

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

            printf("%s trial %zu info=%d status=%d\n", run_ref ? "ref" : "dev", i, info_host, status);

            if(!output)
            {
                output = std::make_unique<rocblas_double_complex[]>(
                    kkrmat.getInMemDataSize() / sizeof(rocblas_double_complex));
                // copy results back
                if(hipMemcpy(output.get(), kkrmat_data_device.data(), kkrmat.getInMemDataSize(),
                             hipMemcpyDeviceToHost)
                   != hipSuccess)
                    throw std::runtime_error("failed to memcpy output");

                if(!run_ref)
                {
                    // also check convergence of dev solution while we're here

                    // original kkr is A
                    auto& A = kkrmat_data_device;
                    // reconstruct B from padded tmat
                    gpubuf_t<rocblas_double_complex> B;
                    if(B.alloc(tmat_data_device.size()) != hipSuccess)
                        throw std::runtime_error("failed to alloc B");
                    if(hipMemcpy(B.data(), tmat_data_pad_host.get(), B.size(), hipMemcpyHostToDevice)
                       != hipSuccess)
                        throw std::runtime_error("failed to re-memcpy to device");

                    // X was written to tmat
                    auto& X = tmat_data_device;

                    // compute residual: R = B - A * X, store it in B
                    const rocblas_double_complex alpha{-1, 0};
                    const rocblas_double_complex beta{1, 0};
                    auto status = rocblas_zgemm(
                        handle, rocblas_operation_none, rocblas_operation_none, kkrmat_dims[0],
                        tmat_dims[0], kkrmat_dims[1], &alpha, A.data(), kkrmat_dims[0], X.data(),
                        kkrmat_dims[0], &beta, B.data(), kkrmat_dims[0]);

                    if(status != rocblas_status_success)
                    {
                        throw std::runtime_error("gemm failed");
                    }

                    // compute norm
                    auto residual_host
                        = std::make_unique<rocblas_double_complex[]>(kkrmat_dims[0] * kkrmat_dims[1]);
                    if(hipMemcpy(residual_host.get(), B.data(), B.size(), hipMemcpyDeviceToHost)
                       != hipSuccess)
                        throw std::runtime_error("failed to copy residual back");

                    double l_inf_r = 0.0;
                    double l_2_r = 0.0;
#pragma omp parallel for reduction(max : l_inf_r) reduction(+ : l_2_r)
                    for(size_t i = 0; i < kkrmat_dims[0] * kkrmat_dims[1]; ++i)
                    {
                        double rdiff = std::abs(residual_host[i].x);
                        l_inf_r = std::max(rdiff, l_inf_r);
                        double idiff = std::abs(residual_host[i].y);
                        l_inf_r = std::max(idiff, l_inf_r);
                        l_2_r += rdiff * rdiff + idiff * idiff;
                    }
                    l_2_r = sqrt(l_2_r);
                    printf("Residual L2=%e, L-inf=%e\n", l_2_r, l_inf_r);

                    double l_inf_b = 0.0;
                    double l_2_b = 0.0;
#pragma omp parallel for reduction(max : l_inf_b) reduction(+ : l_2_b)
                    for(size_t i = 0; i < kkrmat_dims[0] * tmat_dims[0]; ++i)
                    {
                        double rdiff = std::abs(tmat_data_pad_host[i].x);
                        l_inf_r = std::max(rdiff, l_inf_r);
                        double idiff = std::abs(tmat_data_pad_host[i].y);
                        l_inf_r = std::max(idiff, l_inf_r);
                        l_2_r += rdiff * rdiff + idiff * idiff;
                    }
                    l_2_r = sqrt(l_2_r);
                    printf("B L2=%e, B=%e\n", l_2_r, l_inf_r);

                    printf("R/B L2=%e, B=%e\n", l_2_b / l_2_r, l_inf_b / l_inf_r);
                }
            }
        }
    }

    rocblas_destroy_handle(handle);
    handle = nullptr;

    // compare results
    double l_inf = 0.0;
    double l_2 = 0.0;
#pragma omp parallel for reduction(max : l_inf) reduction(+ : l_2)
    for(size_t i = 0; i < kkrmat.getInMemDataSize() / sizeof(rocblas_double_complex); ++i)
    {
        double rdiff = std::abs(ref_output[i].x - dev_output[i].x);
        l_inf = std::max(rdiff, l_inf);
        double idiff = std::abs(ref_output[i].y - dev_output[i].y);
        l_inf = std::max(idiff, l_inf);
        l_2 += rdiff * rdiff + idiff * idiff;
    }
    l_2 = sqrt(l_2);
    printf("Diff L2=%e, L-inf=%e\n", l_2, l_inf);

    for(auto run_ref : {true, false})
    {
        auto& gpu_time = run_ref ? ref_gpu_time : dev_gpu_time;
        auto name = run_ref ? "Ref" : "Dev";

        printf("%s execution times (ms):", name);
        for(auto t : gpu_time)
        {
            printf(" %.2f", static_cast<double>(t));
        }

        std::sort(gpu_time.begin(), gpu_time.end());
        auto median = ntrial % 2 ? gpu_time[ntrial / 2]
                                 : (gpu_time[ntrial / 2] + gpu_time[ntrial / 2 + 1]) / 2;
        printf("\n%s median time (ms): %.2f\n", name, median);
    }

    return 0;
}
