#include <H5Cpp.h>

#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "gpubuf.h"
#include "rocsolver/rocsolver.h"

int main()
{
  // define double complex type
  H5::CompType ztype(sizeof(double)*2);
  ztype.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
  ztype.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);

  // load files
  H5::H5File kkr("kkrmat_000.h5", H5F_ACC_RDONLY);
  H5::H5File t("tmat_000.h5", H5F_ACC_RDONLY);

  auto kkrmat = kkr.openDataSet("kkrmat");
  auto tmat = t.openDataSet("tmat");

  printf("kkr size = %zu\nt size = %zu\n",
         kkrmat.getInMemDataSize(),
         tmat.getInMemDataSize());

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

  // copy data to host memory
  auto kkrmat_data_host = std::make_unique<rocblas_double_complex[]>(kkrmat.getInMemDataSize() / sizeof(rocblas_double_complex));
  auto tmat_data_host = std::make_unique<rocblas_double_complex[]>(tmat.getInMemDataSize() / sizeof(rocblas_double_complex));

  kkrmat.read(kkrmat_data_host.get(), ztype);
  printf("read kkr successfully\n");
  tmat.read(tmat_data_host.get(), ztype);
  printf("read tmat successfully\n");

  gpubuf_t<rocblas_double_complex> kkrmat_data_device;
  gpubuf_t<rocblas_double_complex> tmat_data_device;
  if(kkrmat_data_device.alloc(kkrmat.getInMemDataSize()) != hipSuccess)
    throw std::runtime_error("failed to hipmalloc kkr");
  if(tmat_data_device.alloc(tmat.getInMemDataSize()) != hipSuccess)
    throw std::runtime_error("failed to hipmalloc t");

  // copy to device
  if(hipMemcpy(kkrmat_data_device.data(), kkrmat_data_host.get(), kkrmat.getInMemDataSize(), hipMemcpyHostToDevice) !=hipSuccess)
    throw std::runtime_error("failed to memcpy kkr to device");
  if(hipMemcpy(tmat_data_device.data(), tmat_data_host.get(), tmat.getInMemDataSize(), hipMemcpyHostToDevice) !=hipSuccess)
    throw std::runtime_error("failed to memcpy t to device");
  
  gpubuf_t<rocblas_int> ipiv;
  if(ipiv.alloc(sizeof(rocblas_int)*kkrmat_dims[0]) != hipSuccess)
    throw std::runtime_error("failed to hipmalloc ipiv");

  gpubuf_t<rocblas_int> info;
  if(info.alloc(sizeof(rocblas_int)) != hipSuccess)
    throw std::runtime_error("failed to hipmalloc info");
  
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // solve
  auto status = rocsolver_zgesv(handle,
                                kkrmat_dims[0],
                                tmat_dims[0],
                                kkrmat_data_device.data(),
                                kkrmat_dims[1],
                                ipiv.data(),
                                tmat_data_device.data(),
                                tmat_dims[1],
                                info.data());

  rocblas_int info_host;
  if(hipMemcpy(&info_host, info.data(), sizeof(rocblas_int), hipMemcpyDeviceToHost) != hipSuccess)
    throw std::runtime_error("failed to copy info back");
  
  printf("info:%d, status: %d\n", info_host, status);
  return 0;
}
