/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once
#include <type_traits>
#include <typeinfo>

#include "hip/hip_bf16.h"
#include "hip/hip_bfloat16.h"
#include "hip/hip_fp16.h"

#include "rocblas.hpp"
#include "roclapack_getf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

#include "roclapack_getrf.hpp"

#include "auxiliary/rocauxiliary_complex2reim.hpp"
#include "auxiliary/rocauxiliary_gemm_ex.hpp"

ROCSOLVER_BEGIN_NAMESPACE

static bool constexpr use_out_of_place = true;

// -------------------------------------------------------------------
// compute   scaling_array[i] = dlimit/amax_array[i], i=0:(n-1)
//
// launch as dim(nbx,1,1), dim(nx,1,1),  where nbx = ceil( n, nx )
// -------------------------------------------------------------------
template <typename Treal, typename I>
static __device__ void gen_scaling_kernel(I const n,
                                          Treal const dlimit,

                                          Treal* const amax_array,
                                          Treal* const scaling_array)
{
    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    for(I i = i_start; i < n; i += i_inc)
    {
        auto const amax = amax_array[i];
        auto const inv_amax = (amax == 0) ? 1 : 1.0 / amax;
        scaling_array[i] = dlimit * inv_amax;
    }
}

template <typename Treal, typename I>
static void gen_scaling(hipStream_t stream,
                        I const n,
                        Treal const dlimit,

                        Treal* const amax_array,
                        Treal* const scaling_array)
{
    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const nx = 256;
    I const nbx = ceil(n, nx);
    gen_scaling_kernel<Treal, I>
        <<<dim3(nbx, 1, 1), dim3(nx, 1, 1), 0, stream>>>(n, dlimit, amax_array, scaling_array);
}

// ---------------------------------------------------------
// the block size may be a tuning parameter for optimization
// ---------------------------------------------------------
template <bool ISBATCHED, typename T, typename I>
static I getrf_mxp_get_blksize(I const n, bool const use_pivot)
{
    return (std::min(n, 1024));
}

/** Return the sizes of the different workspace arrays **/
template <typename T, typename Treduced, typename I>
void rocsolver_getrf_mxp_getMemorySize(const I m,
                                       const I n,
                                       const bool pivot,
                                       const I batch_count,

                                       size_t* p_size_work,
                                       const I lda = 1,
                                       const I inca = 1)
{
    bool constexpr BATCHED = false;
    bool constexpr STRIDED = false;

    *p_size_work = 0;

    // if quick return, no need of workspace
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    size_t size_work = 0;

    bool constexpr is_fp16 = std::is_same<Treduced, rocblas_half>::value
        || std::is_same<Treduced, __half>::value || std::is_same<Treduced, _Float16>::value;

    bool constexpr is_complex = rocblas_is_complex<T>;
    bool constexpr is_batched = (BATCHED || STRIDED);
    bool constexpr ISBATCHED = is_batched;

    using S = decltype(std::real(T{}));
    using Smax = S;

    if(is_batched)
    {
        // work space for rocblas GEMM
        size_t const size_ptr_array = sizeof(T*) * batch_count * 3;
        size_work += size_ptr_array;
    }

    auto const min_mn = std::min(m, n);
    auto const dim = min_mn;
    I const blk = getrf_mxp_get_blksize<ISBATCHED, T>(dim, pivot);

    {
        // ----------------
        // memory for getrf
        // ----------------

        bool optim_mem = true;
        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        size_t size_pivotval = 0;
        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;

        auto const nn = min_mn;
        rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T, I>(
            m, nn, pivot, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
            &size_work4, &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        size_t const size_getrf = size_scalars + size_work1 + size_work2 + size_work3 + size_work4
            + size_pivotval + size_pivotidx + size_iipiv + size_iinfo;

        size_work += size_getrf;
    }

    // --------------------------------
    // space for rocsolverCall_gemm_strided_batched_ex
    // --------------------------------

    {
        size_t size_gemm_ex = 0;
        rocblas_operation const trans_A = rocblas_operation_none;
        rocblas_operation const trans_B = rocblas_operation_none;

        rocblasCall_gemm_strided_batched_ex_getMemorySize<T, Treduced, I>(
            trans_A, trans_B, m, n, blk, batch_count, &size_gemm_ex);
        size_work += size_gemm_ex;
    }

    *p_size_work = size_work;
}

#ifndef CHECK_MEM
#define CHECK_MEM(pfree)                                          \
    {                                                             \
        bool const is_memory_ok = (pfree <= (pwork + size_work)); \
        assert(is_memory_ok);                                     \
        if(!is_memory_ok)                                         \
        {                                                         \
            return (rocblas_status_internal_error);               \
        }                                                         \
    }
#endif

#ifndef ROCBLAS_CHECK
#define ROCBLAS_CHECK(fcn)                  \
    {                                       \
        auto const istat = (fcn);           \
        if(istat != rocblas_status_success) \
        {                                   \
            return (istat);                 \
        }                                   \
    }
#endif

template <typename T, typename Treduced, typename I, typename Istride, typename INFO>
rocblas_status rocsolver_getrf_mxp_template(rocblas_handle handle,
                                            I const m,
                                            I const n,

                                            T* const A,
                                            Istride const shiftA,
                                            I const inca,
                                            I const lda,
                                            Istride const strideA,

                                            I* const ipiv,
                                            Istride const shiftP,
                                            Istride const strideP,

                                            INFO* const info,
                                            I const batch_count,
                                            bool const pivot,

                                            void* const work,
                                            size_t const size_work)
{
    ROCSOLVER_ENTER("getrf_mxp", "m:", m, "n:", n, "shiftA:", shiftA, "inca:", inca, "lda:", lda,
                    "shiftP:", shiftP, "bc:", batch_count);

    bool constexpr BATCHED = false;
    bool constexpr STRIDED = false;

    bool constexpr is_complex = rocblas_is_complex<T>;

    using S = decltype(std::real(T{}));
    using Smax = S;

    I const ldA = lda;
    I const fp16_max = 65504; // max valid representable value in FP16
    I const fp16_max_m1 = fp16_max - 1;

    bool constexpr is_fp16 = std::is_same<Treduced, __half>::value
        || std::is_same<Treduced, _Float16>::value || std::is_same<Treduced, rocblas_half>::value;

    Smax const dlimit = (is_fp16) ? fp16_max_m1 : 1;

    // quick return
    if(batch_count == 0)
    {
        return rocblas_status_success;
    }

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = (BATCHED || STRIDED);
    I dim = std::min(m, n);
    I blocks = 0, blocksy = 0;

    // ---------------
    // reset info array
    // ---------------
    {
        I const nthreads = 64;
        I const blocks = ceil(batch_count, nthreads);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocks, 1, 1), dim3(nthreads, 1, 1), 0, stream,
                                info, batch_count, 0);
    }

    // quick return if no dimensions
    if(m == 0 || n == 0)
    {
        return rocblas_status_success;
    }

    // size of outer blocks
    I blk = getrf_mxp_get_blksize<ISBATCHED, T>(dim, pivot);

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    // -----------------------------------------
    // scratch arrays for getrf LU factorization
    // -----------------------------------------
    T* scalars = nullptr;
    void* work1 = nullptr;
    void* work2 = nullptr;
    void* work3 = nullptr;
    void* work4 = nullptr;
    T* pivotval = nullptr;
    I* pivotidx = nullptr;
    INFO* iinfo = nullptr;
    I* iipiv = nullptr;

    {
        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        size_t size_pivotval = 0;
        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;

        bool optim_mem = true;

        I const min_mn = std::min(m, n);
        I const dim = min_mn;
        I const nn = dim;
        rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T, I>(

            m, nn, pivot, batch_count,

            &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4, &size_pivotval,
            &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        scalars = (T*)pfree;
        pfree += size_scalars;
        work1 = (void*)pfree;
        pfree += size_work1;
        work2 = (void*)pfree;
        pfree += size_work2;
        work3 = (void*)pfree;
        pfree += size_work3;
        work4 = (void*)pfree;
        pfree += size_work4;

        pivotval = (T*)pfree;
        pfree += size_pivotval;
        pivotidx = (I*)pfree;
        pfree += size_pivotidx;
        iipiv = (I*)pfree;
        pfree += size_iipiv;
        iinfo = (INFO*)pfree;
        pfree += size_iinfo;

        CHECK_MEM(pfree);
    }

    if(blk == 0)
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, inca, lda, strideA,
                                                      ipiv, shiftP, strideP, info, batch_count,
                                                      scalars, pivotval, pivotidx, pivot);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T one = 1;
    T minone = -1;

    I jb, dimx, dimy;
    I nextpiv, mm, nn;
    size_t lmemsize;
    I j = 0;
    bool const optim_mem = true;

    // in the npvt cases, panel determines whether the whole block-panel or only the
    // diagonal block is factorized
    bool panel = false;
    if(blk < 0)
    {
        panel = true;
        blk = -blk;
    }

    // MAIN LOOP
    for(I j = 0; j < dim; j += blk)
    {
        jb = std::min(dim - j, blk);

        if(pivot || panel)
        {
            // factorize outer block panel
            getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j, jb, n, A, shiftA + j * inca, inca,
                                               lda, strideA, ipiv, shiftP + j, strideP, info,
                                               batch_count, pivot, scalars, work1, work2, work3,
                                               work4, optim_mem, pivotval, pivotidx, j, iipiv, m);
        }
        else
        {
            // factorize only outer diagonal block
            getrf_panelLU<BATCHED, STRIDED, T>(handle, jb, jb, n, A, shiftA + j * inca, inca, lda,
                                               strideA, ipiv, shiftP + j, strideP, info,
                                               batch_count, pivot, scalars, work1, work2, work3,
                                               work4, optim_mem, pivotval, pivotidx, j, iipiv, m);

            // update remaining rows in outer panel
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
                m - j - jb, jb, A, shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
                shiftA + idx2D(jb + j, j, inca, lda), inca, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);
        }

        // update trailing matrix
        nextpiv = j + jb; //position for the matrix update
        mm = m - nextpiv; //size for the matrix update
        nn = n - nextpiv; //size for the matrix update
        if(nextpiv < n)
        {
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb, nn, A,
                shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
                shiftA + idx2D(j, nextpiv, inca, lda), inca, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);

            if(nextpiv < m)
            {
                bool const use_regular_gemm = std::is_same<T, Treduced>::value;
                rocblas_status istat = rocblas_status_success;

                if(use_regular_gemm)
                {
                    // -----------------------------------------------------------
                    // Mixed precision is not required, use regular GEMM in rocblas
                    // -----------------------------------------------------------
                    istat = rocsolver_gemm(
                        handle, rocblas_operation_none, rocblas_operation_none, mm, nn, jb, &minone,

                        // L21
                        A, shiftA + idx2D(nextpiv, j, inca, lda), inca, lda, strideA,

                        // U12
                        A, shiftA + idx2D(j, nextpiv, inca, lda), inca, lda, strideA,

                        &one,

                        // A22
                        A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), inca, lda, strideA,

                        batch_count, (T**)nullptr);
                }
                else
                {
                    rocblas_operation const trans_A = rocblas_operation_none;
                    rocblas_operation const trans_B = rocblas_operation_none;

                    T alpha = minone;
                    T beta = one;

                    rocblas_datatype const type_A = rocblas_datatype_from_type<T>;
                    rocblas_datatype const type_B = type_A;
                    rocblas_datatype const type_C = type_A;
                    rocblas_datatype const type_D = type_C;

                    rocblas_datatype const compute_type = rocblas_datatype_from_type<Treduced>;

                    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
                    int32_t solution_index = 0;
                    uint32_t flags = rocblas_gemm_flags_none;

                    auto const L21 = A + shiftA + idx2D(nextpiv, j, inca, lda);
                    auto const ldL21 = lda;
                    auto const stride_L21 = strideA;

                    auto const U12 = A + shiftA + idx2D(j, nextpiv, inca, lda);
                    auto const ldU12 = lda;
                    auto const stride_U12 = strideA;

                    auto const A22 = A + shiftA + idx2D(nextpiv, nextpiv, inca, lda);
                    auto const ldA22 = lda;
                    auto const stride_A22 = strideA;

                    size_t const size_remain = (pwork + size_work) - pfree;

                    auto const istat = rocblasCall_gemm_strided_batched_ex(
                        handle,

                        trans_A, trans_B, mm, nn, jb,

                        &alpha,

                        L21, type_A, ldL21, stride_L21,

                        U12, type_B, ldU12, stride_U12,

                        &beta,

                        A22, type_C, ldA22, stride_A22,

                        A22, type_D, ldA22, stride_A22,

                        batch_count,

                        compute_type, algo, solution_index, flags,

                        (void*)pfree, size_remain);
                }
                if(istat != rocblas_status_success)
                {
                    rocblas_set_pointer_mode(handle, old_mode);
                    return (istat);
                }
            }
        }
    } // end for j

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
