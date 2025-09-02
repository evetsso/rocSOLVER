/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "lib_host_helpers.hpp"

#include "auxiliary/rocauxiliary_complex2reim.hpp"
#include "auxiliary/rocauxiliary_lacpy.hpp"
#include "lapack/roclapack_gesv_ex.hpp"

#include <limits>

ROCSOLVER_BEGIN_NAMESPACE

// XXX: this shouldn't do scope magic
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

template <typename Tfull, typename Treduced, typename I>
static void rocblasCall_gemm_strided_batched_ex_getMemorySize(rocblas_operation const trans_A,
                                                              rocblas_operation const trans_B,
                                                              I const m,
                                                              I const n,
                                                              I const k,
                                                              I const batch_count,
                                                              size_t* p_size_work)
{
    using Sf = decltype(std::real(Tfull{}));
    using Sr = decltype(std::real(Treduced{}));

    *p_size_work = 0;
    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    // ---------------------------
    // may need array of pointers
    // ---------------------------
    size_t size_work = sizeof(Tfull*) * batch_count;

    bool const is_none_A = (trans_A == rocblas_operation_none);
    bool const is_none_B = (trans_B == rocblas_operation_none);

    I const nrows_A = (is_none_A) ? m : k;
    I const ncols_A = (is_none_A) ? k : m;

    I const nrows_B = (is_none_B) ? k : n;
    I const ncols_B = (is_none_B) ? n : k;

    I const nrows_C = m;
    I const ncols_C = n;

    rocblas_datatype constexpr r_type = rocblas_datatype_from_type<Treduced>;
    rocblas_datatype constexpr f_type = rocblas_datatype_from_type<Tfull>;

    bool constexpr is_fp16_compute
        = (r_type == rocblas_datatype_f16_c) || (r_type == rocblas_datatype_f16_r);

    bool constexpr is_bf16_compute
        = (r_type == rocblas_datatype_bf16_c) || (r_type == rocblas_datatype_bf16_r);

    bool constexpr is_fp32_store
        = (f_type == rocblas_datatype_f32_c) || (f_type == rocblas_datatype_f32_r);

    // -------------------------------------------
    // only process the special case for FP32 storage type and
    // reduced precision  FP16 or BF16 as compute type
    // -------------------------------------------
    bool const need_temp_storage = (is_fp32_store && (is_fp16_compute || is_bf16_compute));
    bool const is_complex = (f_type == rocblas_datatype_f32_c);

    if(need_temp_storage)
    {
        size_t const size_A_re_chop = sizeof(Treduced) * nrows_A * ncols_A * batch_count;
        size_t const size_B_re_chop = sizeof(Treduced) * nrows_B * ncols_B * batch_count;

        size_work += size_A_re_chop;
        size_work += size_B_re_chop;

        bool constexpr need_amax = false;
        // --------------------------------
        // storage for amax to fit in FP16
        // --------------------------------
        if(is_fp16_compute && need_amax)
        {
            size_t size_amax_A_re = 0;
            size_t size_amax_A_im = 0;
            size_t size_amax_B_re = 0;
            size_t size_amax_B_im = 0;

            size_amax_A_re = sizeof(Sr) * batch_count;
            size_amax_B_re = sizeof(Sr) * batch_count;

            size_work += size_amax_A_re;
            size_work += size_amax_B_re;

            if(is_complex)
            {
                size_amax_A_im = size_amax_A_re;
                size_amax_B_im = size_amax_B_re;

                size_work += size_amax_A_im;
                size_work += size_amax_B_im;
            }
        }

        if(is_complex)
        {
            size_t const size_A_im_chop = size_A_re_chop;
            size_t const size_B_im_chop = size_B_re_chop;

            size_work += size_A_im_chop;
            size_work += size_B_im_chop;
        }

        if(is_complex)
        {
            // ---------------------------------------------------
            // need to split complex C matrix
            // into the real part matrix and imaginary part matrix
            // ---------------------------------------------------
            size_t const size_C_re = sizeof(Sf) * nrows_C * ncols_C * batch_count;
            size_t const size_C_im = size_C_re;

            size_work += size_C_re;
            size_work += size_C_im;
        }
    }

    *p_size_work = size_work;
}

//
// simple interface to emulate rocblas_gemm_ex() or rocblas_gemm_ex_strided_batched()
//
template <typename Tfull, typename TCompute, typename I, typename Istride>
static rocblas_status rocblasCall_gemm_strided_batched_ex_impl(rocblas_handle handle,

                                                               rocblas_operation const trans_A,
                                                               rocblas_operation const trans_B,

                                                               I const m,
                                                               I const n,
                                                               I const k,

                                                               Tfull* const alpha,

                                                               Tfull* const A,
                                                               I const ld_A,
                                                               Istride const stride_A,

                                                               Tfull* const B,
                                                               I const ld_B,
                                                               Istride const stride_B,

                                                               Tfull* const beta,

                                                               Tfull* C,
                                                               I const ld_C,
                                                               Istride const stride_C,

                                                               Tfull* D,
                                                               I const ld_D,
                                                               Istride const stride_D,

                                                               I const batch_count,

                                                               rocblas_gemm_algo algo,
                                                               int32_t solution_index,
                                                               uint32_t flags,

                                                               void* work,
                                                               size_t size_work)
{
#if(0)
    // A, B, C, D must be all real or all complex
    bool const is_all_same_type = (type_A == type_B) && (type_B == type_C) && (type_C == type_D);
    if(!is_all_same_type)
    {
        return (rocblas_status_not_implemented);
    }

    // ----------------------------------------
    // try evaluation as one of the supported types
    // ----------------------------------------
    auto const istat = (rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k, alpha,

                                                        A_arg, type_A, ld_A, stride_A,

                                                        B_arg, type_B, ld_B, stride_B,

                                                        beta,

                                                        C_arg, type_C, ld_C, stride_C,

                                                        D_arg, type_D, ld_D, stride_D,

                                                        batch_count,

                                                        compute_type, algo, solution_index, flags));
    if(istat != rocblas_status_not_implemented)
    {
        return (istat);
    }

    Tfull* const A = (Tfull*)A_arg;
    Tfull* const B = (Tfull*)B_arg;
    Tfull* const C = (Tfull*)C_arg;
    Tfull* const D = (Tfull*)D_arg;

    // ----------------------------------------------------
    // implement computation where storage type is
    // F32_C or F32_R and compute type is BF16 or FP16
    //
    // This is to match cublasGemmEx
    // with A/B/C type be CUDA_R_32F or CUDA_C_32F
    // but compute type is
    // CUBLAS_COMPUTE_F32_FAST_16BF
    // or
    // CUBLAS_COMPUTE_F32_FAST_16F
    // ----------------------------------------------------

    bool const is_fp32_store
        = (type_C == rocblas_datatype_f32_r) || (type_C == rocblas_datatype_f32_c);

    bool const is_fp16_compute
        = (compute_type == rocblas_datatype_f16_r) || (compute_type == rocblas_datatype_f16_c);

    bool const is_bf16_compute
        = (compute_type == rocblas_datatype_bf16_r) || (compute_type == rocblas_datatype_bf16_c);

    bool const is_supported = is_fp32_store && (is_fp16_compute || is_bf16_compute);
    if(!is_supported)
    {
        return rocblas_status_not_implemented;
    }
#endif
    bool constexpr is_complex = rocblas_is_complex<Tfull>;
    // ---------------------------------------
    // NOTE we assume alpha and beta don't have
    // imaginary part
    // ---------------------------------------
    if(is_complex)
    {
        Tfull* const p_alpha = (Tfull*)alpha;
        Tfull* const p_beta = (Tfull*)beta;
        bool is_ok = (std::imag(*p_alpha) == 0) && (std::imag(*p_beta) == 0);
        if(!is_ok)
        {
            return (rocblas_status_not_implemented);
        }
    }

    // =============================== //

    bool const is_none_A = (trans_A == rocblas_operation_none);
    bool const is_none_B = (trans_B == rocblas_operation_none);

    I const nrows_A = (is_none_A) ? m : k;
    I const ncols_A = (is_none_A) ? k : m;

    I const nrows_B = (is_none_B) ? k : n;
    I const ncols_B = (is_none_B) ? n : k;

    I const nrows_C = m;
    I const ncols_C = n;

    using Sf = decltype(std::real(Tfull{}));
    using Sr = decltype(std::real(TCompute{}));

    {
        // -------------------------------------------
        // double check the types Tfull and TCompute are
        // consistent with the compute_type value
        // -------------------------------------------
        bool is_ok = std::is_same<Sf, float>::value && (sizeof(Sr) == 2);
        if(!is_ok)
        {
            return (rocblas_status_internal_error);
        }

        // -------------
        // assume D == C
        // -------------
        bool const is_D_same_as_C = (D == C);
        if(!is_D_same_as_C)
        {
            return (rocblas_status_not_implemented);
        }
    }

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    bool const is_fp16_compute = std::is_same<TCompute, rocblas_float>::value
        || std::is_same<TCompute, rocblas_complex_num<rocblas_half>>::value;

    double const fp32_max = std::numeric_limits<float>::max();
    double const bf16_max = fp32_max;
    double const fp16_max = 65504; // largest valid number in FP16
    double const dlimit = is_fp16_compute ? fp16_max : bf16_max;

    Sf* C_re = nullptr;
    Sf* C_im = nullptr;

    Sr* A_re_chop = nullptr;
    Sr* A_im_chop = nullptr;
    Sr* B_re_chop = nullptr;
    Sr* B_im_chop = nullptr;

    I const ldA_re_chop = nrows_A;
    I const ldA_im_chop = ldA_re_chop;

    Istride stride_A_re_chop = ldA_re_chop * ncols_A;
    Istride stride_A_im_chop = stride_A_re_chop;

    I const ldB_re_chop = nrows_B;
    I const ldB_im_chop = ldB_re_chop;

    Istride stride_B_re_chop = ldB_re_chop * ncols_B;
    Istride stride_B_im_chop = stride_B_re_chop;

    size_t const size_A_re_chop = sizeof(Sr) * stride_A_re_chop * batch_count;
    size_t const size_B_re_chop = sizeof(Sr) * stride_B_re_chop * batch_count;

    size_t const size_A_im_chop = (is_complex) ? size_A_re_chop : 0;
    size_t const size_B_im_chop = (is_complex) ? size_B_re_chop : 0;

    I const ldC = ld_C;
    I const ldC_re = (is_complex) ? nrows_C : ld_C;
    I const ldC_im = (is_complex) ? nrows_C : 0;

    Istride const shift_C_re = 0;
    Istride const shift_C_im = 0;

    Istride const stride_C_re = (is_complex) ? ldC_re * ncols_C : stride_C;
    Istride const stride_C_im = (is_complex) ? ldC_im * ncols_C : 0;

    size_t const size_C_re = sizeof(Sf) * stride_C_re * batch_count;
    size_t const size_C_im = size_C_re;

    rocblas_datatype const A_re_type = rocblas_datatype_from_type<Sr>;
    rocblas_datatype const A_im_type = A_re_type;

    rocblas_datatype const B_re_type = rocblas_datatype_from_type<Sr>;
    rocblas_datatype const B_im_type = B_re_type;

    rocblas_datatype const C_re_type = rocblas_datatype_from_type<Sf>;
    rocblas_datatype const C_im_type = C_re_type;

    // ----------------------------------------------------
    // assume everything is executed with scalars on the host
    // ----------------------------------------------------
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    bool const is_valid_mode = (old_mode == rocblas_pointer_mode_host);
    if(!is_valid_mode)
    {
        return (rocblas_status_internal_error);
    }

    // --------------------------------------------
    // split into the real part and imaginary part
    // in reduced precision
    // --------------------------------------------
    A_re_chop = (Sr*)pfree;
    pfree += size_A_re_chop;
    B_re_chop = (Sr*)pfree;
    pfree += size_B_re_chop;
    if constexpr(is_complex)
    {
        A_im_chop = (Sr*)pfree;
        pfree += size_A_im_chop;
        B_im_chop = (Sr*)pfree;
        pfree += size_B_im_chop;
    }

    CHECK_MEM(pfree);

    Istride const shift_A = 0;
    Istride const shift_B = 0;
    Istride const shift_C = 0;

    Istride const shift_A_re_chop = 0;
    Istride const shift_A_im_chop = 0;

    Istride const shift_B_re_chop = 0;
    Istride const shift_B_im_chop = 0;

    if constexpr(is_complex)
    {
        double* const amax_re = nullptr;
        double* const amax_im = nullptr;

        complex2reim_outofplace(handle, nrows_A, ncols_A,

                                A, shift_A, ld_A, stride_A,

                                A_re_chop, shift_A_re_chop, ldA_re_chop, stride_A_re_chop,

                                A_im_chop, shift_A_im_chop, ldA_im_chop, stride_A_im_chop,

                                batch_count,

                                dlimit, amax_re, amax_im);

        complex2reim_outofplace(handle, nrows_B, ncols_B,

                                B, shift_B, ld_B, stride_B,

                                B_re_chop, shift_B_re_chop, ldB_re_chop, stride_B_re_chop,

                                B_im_chop, shift_B_im_chop, ldB_im_chop, stride_B_im_chop,

                                batch_count,

                                dlimit, amax_re, amax_im);
    }
    else
    {
        char const uplo = 'A';

        ROCBLAS_CHECK(rocsolver_lacpy_template(handle, uplo, nrows_A, ncols_A,

                                               A, shift_A, ld_A, stride_A,

                                               A_re_chop, shift_A_re_chop, ldA_re_chop,
                                               stride_A_re_chop,

                                               batch_count));

        ROCBLAS_CHECK(rocsolver_lacpy_template(handle, uplo, nrows_B, ncols_B,

                                               B, shift_B, ld_B, stride_B,

                                               B_re_chop, shift_B_re_chop, ldB_re_chop,
                                               stride_B_re_chop,

                                               batch_count));
    }

    // ----------------------------------------------
    // split the "C" into the real part and imag part
    // ----------------------------------------------
    if constexpr(is_complex)
    {
        double const dlimit_C = fp32_max;
        double* const amax_re = nullptr;
        double* const amax_im = nullptr;

        C_re = (Sf*)pfree;
        pfree += size_C_re;
        C_im = (Sf*)pfree;
        pfree += size_C_im;

        CHECK_MEM(pfree);

        complex2reim_outofplace(handle, nrows_C, ncols_C,

                                C, shift_C, ldC, stride_C,

                                C_re, shift_C_re, ldC_re, stride_C_re,

                                C_im, shift_C_im, ldC_im, stride_C_im,

                                batch_count,

                                dlimit_C, amax_re, amax_im);
    }
    else
    {
        C_re = (Sf*)C;
    }

    // ------------------------------------
    // (Cr + J * Cc) = beta * (Cr + J * Cc) +
    //              alpha * ( Ar + J * Ac ) * (Br + J * Bc)
    //
    // Cr = beta * Cr + alpha * { Ar * Br - Ac * Bc }
    // Cc = beta * Cc + alpha * { Ac * Br + Ar * Bc }
    // ------------------------------------

    // ------------------------------------
    // (1) Cr = beta * Cr + alpha * Ar * Br
    // (2) Cr =        Cr - alpha * Ac * Bc
    //
    // (3) Cc = beta * Cc + alpha * Ac * Br
    // (4) Cc =        Cc + alpha * Ar * Bc
    // ------------------------------------

    // -----------------------------------------
    // step (1) Cr = beta * Cr + alpha * Ar * Br
    // -----------------------------------------

    // -----------------------------------
    // NOTE: raise the compute type to FP32
    // when calling rocblas_gemm_ex()
    // for higher accuracy and
    // the output array is in FP32
    // -----------------------------------

    rocblas_datatype lcompute_type = rocblas_datatype_f32_r;
    Tfull* const p_alpha = (Tfull*)alpha;
    Tfull* const p_beta = (Tfull*)beta;

    const float alpha_f32 = std::real(*p_alpha);
    const float beta_f32 = std::real(*p_beta);
    const float beta_one_f32 = 1.0;

    {
        // ---------------------------------------------
        // no need to adjust alpha or beta
        // ---------------------------------------------

        ROCBLAS_CHECK(
            rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                            &alpha_f32,

                                            A_re_chop, A_re_type, ldA_re_chop, stride_A_re_chop,

                                            B_re_chop, B_re_type, ldB_re_chop, stride_B_re_chop,

                                            &beta_f32,

                                            C_re, C_re_type, ldC_re, stride_C_re,

                                            C_re, C_re_type, ldC_re, stride_C_re, // D matrix

                                            batch_count,

                                            lcompute_type, algo, solution_index, flags));
    }

    if constexpr(is_complex)
    {
        // ------------------------------------
        // step (2) Cr =        Cr - alpha * Ac * Bc
        // ------------------------------------

        {
            float neg_alpha_f32 = -alpha_f32;

            ROCBLAS_CHECK(
                rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                                &neg_alpha_f32,

                                                A_im_chop, A_im_type, ldA_im_chop, stride_A_im_chop,

                                                B_im_chop, B_im_type, ldB_im_chop, stride_B_im_chop,

                                                &beta_one_f32,

                                                C_re, C_re_type, ldC_re, stride_C_re,

                                                C_re, C_re_type, ldC_re, stride_C_re, // D matrix

                                                batch_count,

                                                lcompute_type, algo, solution_index, flags));
        }

        // ------------------------------------
        // step (3) Cc = beta * Cc + alpha * Ar * Bc
        // ------------------------------------

        {
            ROCBLAS_CHECK(
                rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                                &alpha_f32,

                                                A_re_chop, A_re_type, ldA_re_chop, stride_A_re_chop,

                                                B_im_chop, B_im_type, ldB_im_chop, stride_B_im_chop,

                                                &beta_f32,

                                                C_im, C_im_type, ldC_im, stride_C_im,

                                                C_im, C_im_type, ldC_im, stride_C_im, // D matrix

                                                batch_count,

                                                lcompute_type, algo, solution_index, flags));
        }

        // ------------------------------------
        // step (4) Cc =        Cc + alpha * Ac * Br
        // ------------------------------------

        {
            ROCBLAS_CHECK(
                rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k,

                                                &alpha_f32,

                                                A_im_chop, A_im_type, ldA_im_chop, stride_A_im_chop,

                                                B_re_chop, B_re_type, ldB_re_chop, stride_B_re_chop,

                                                &beta_one_f32,

                                                C_im, C_im_type, ldC_im, stride_C_im,

                                                C_im, C_im_type, ldC_im, stride_C_im, // D matrix

                                                batch_count,

                                                lcompute_type, algo, solution_index, flags));
        }

        // -------------------------------------------------------
        // convert from real and imag parts back to complex matrix
        // -------------------------------------------------------

        {
            reim2complex_outofplace(handle, nrows_C, ncols_C,

                                    C_re, shift_C_re, ldC_re, stride_C_re,

                                    C_im, shift_C_im, ldC_im, stride_C_im,

                                    C, shift_C, ldC, stride_C,

                                    batch_count, 0.0, {}, {});
        }
    }

    return (rocblas_status_success);
}

#if(0)
// currently limited to fp32 complex data for all of A, B, C, D
template <typename TA, typename TB, typename TC, typename TD, typename TCompute>
constexpr bool gemm_ex_accepts
    = std::is_same_v<TA, rocblas_float_complex>&& std::is_same_v<TA, TB>&& std::is_same_v<TA, TC>&&
          std::is_same_v<TA, TD> && !std::is_same_v<TA, TCompute>;

template <typename TA, typename TB, typename TC, typename TD, typename TCompute, typename...>
struct gemm_ex_call
{
    rocblas_status operator()(rocblas_handle handle,

                              rocblas_operation const trans_A,
                              rocblas_operation const trans_B,

                              rocblas_int const m,
                              rocblas_int const n,
                              rocblas_int const k,

                              const void* alpha,

                              const void* A,
                              rocblas_int const ld_A,
                              rocblas_int const stride_A,

                              const void* B,
                              rocblas_int const ld_B,
                              rocblas_int const stride_B,

                              const void* beta,

                              const void* C,
                              rocblas_int const ld_C,
                              rocblas_int const stride_C,

                              void* D,
                              rocblas_int const ld_D,
                              rocblas_int const stride_D,

                              rocblas_int const batch_count,

                              rocblas_gemm_algo algo,
                              int32_t solution_index,
                              uint32_t flags,

                              void* work,
                              size_t size_work)
    {
        if constexpr(gemm_ex_accepts<TA, TB, TC, TD, TCompute>)
        {
            return rocblasCall_gemm_strided_batched_ex_impl<TA, TB, TC, TD, TCompute>(
                handle,

                trans_A, trans_B,

                m, n, k,

                static_cast<const TCompute*>(alpha),

                static_cast<const TA*>(A), ld_A, stride_A,

                static_cast<const TB*>(B), ld_B, stride_B,

                static_cast<const TCompute*>(beta),

                static_cast<const TC*>(C), ld_C, stride_C,

                static_cast<TD*>(D), ld_D, stride_D,

                batch_count,

                algo, solution_index, flags,

                work, size_work);
        }
        return rocblas_status_not_implemented;
    }
};
#endif

template <typename I, typename Istride>
static rocblas_status rocblasCall_gemm_strided_batched_ex(rocblas_handle handle,

                                                          rocblas_operation const trans_A,
                                                          rocblas_operation const trans_B,

                                                          I const m,
                                                          I const n,
                                                          I const k,

                                                          const void* alpha,

                                                          const void* A,
                                                          rocblas_datatype type_A,
                                                          I const ld_A,
                                                          Istride const stride_A,

                                                          const void* B,
                                                          rocblas_datatype type_B,
                                                          I const ld_B,
                                                          Istride const stride_B,

                                                          const void* beta,

                                                          const void* C,
                                                          rocblas_datatype type_C,
                                                          I const ld_C,
                                                          Istride const stride_C,

                                                          void* D,
                                                          rocblas_datatype type_D,
                                                          I const ld_D,
                                                          Istride const stride_D,

                                                          I const batch_count,

                                                          rocblas_datatype compute_type,
                                                          rocblas_gemm_algo algo,
                                                          int32_t solution_index,
                                                          uint32_t flags,

                                                          void* work,
                                                          size_t size_work)
{
    {
        // ----------------------------------------------------
        // try to use rocblas, in case
        // this combination of types is  already supported
        // ----------------------------------------------------
        rocblas_status const status
            = rocblas_gemm_strided_batched_ex(handle, trans_A, trans_B, m, n, k, alpha,

                                              A, type_A, ld_A, stride_A,

                                              B, type_B, ld_B, stride_B,

                                              beta,

                                              C, type_C, ld_C, stride_C,

                                              D, type_D, ld_D, stride_D,

                                              batch_count,

                                              compute_type, algo, solution_index, flags);
        if(status != rocblas_status_not_implemented)
        {
            return status;
        }
    }

    // ==============================================

    // A, B, C, D must be all real or all complex
    bool const is_all_same_type = (type_A == type_B) && (type_B == type_C) && (type_C == type_D);
    if(!is_all_same_type)
    {
        return (rocblas_status_not_implemented);
    }

    // ----------------------------------------------------
    // implement computation where storage type is
    // F32_C or F32_R and compute type is BF16 or FP16
    //
    // This is to match cublasGemmEx
    // with A/B/C type be CUDA_R_32F or CUDA_C_32F
    // but compute type is
    // CUBLAS_COMPUTE_F32_FAST_16BF
    // or
    // CUBLAS_COMPUTE_F32_FAST_16F
    // ----------------------------------------------------

    bool const is_fp32_store
        = (type_C == rocblas_datatype_f32_r) || (type_C == rocblas_datatype_f32_c);

    bool const is_fp16_compute
        = (compute_type == rocblas_datatype_f16_r) || (compute_type == rocblas_datatype_f16_c);

    bool const is_bf16_compute
        = (compute_type == rocblas_datatype_bf16_r) || (compute_type == rocblas_datatype_bf16_c);

    bool const is_supported = is_fp32_store && (is_fp16_compute || is_bf16_compute);
    if(!is_supported)
    {
        return rocblas_status_not_implemented;
    }

    // ------------------------------
    // perform type specific dispatch
    // ------------------------------

#define DISPATCH(TFULL, TCOMPUTE)                                                      \
    {                                                                                  \
        istat = rocblasCall_gemm_strided_batched_ex_impl<TFULL, TCOMPUTE, I, Istride>( \
            handle, trans_A, trans_B, m, n, k,                                         \
                                                                                       \
            (TFULL*)alpha,                                                             \
                                                                                       \
            (TFULL*)A, ld_A, stride_A,                                                 \
                                                                                       \
            (TFULL*)B, ld_B, stride_B,                                                 \
                                                                                       \
            (TFULL*)beta,                                                              \
                                                                                       \
            (TFULL*)C, ld_C, stride_C,                                                 \
                                                                                       \
            (TFULL*)D, ld_D, stride_D,                                                 \
                                                                                       \
            batch_count, algo, solution_index, flags, work, size_work);                \
    }

    rocblas_status istat = rocblas_status_not_implemented;
    if((type_C == rocblas_datatype_f32_c) && (compute_type == rocblas_datatype_f16_r))
    {
        DISPATCH(rocblas_complex_num<float>, rocblas_half);
    }
    else if((type_C == rocblas_datatype_f32_c) && (compute_type == rocblas_datatype_f16_c))
    {
        DISPATCH(rocblas_complex_num<float>, rocblas_complex_num<rocblas_half>);
    }
    else if((type_C == rocblas_datatype_f32_c) && (compute_type == rocblas_datatype_bf16_r))
    {
        DISPATCH(rocblas_complex_num<float>, rocblas_bfloat16);
    }
    else if((type_C == rocblas_datatype_f32_c) && (compute_type == rocblas_datatype_bf16_c))
    {
        DISPATCH(rocblas_complex_num<float>, rocblas_complex_num<rocblas_bfloat16>);
    }
    else if((type_C == rocblas_datatype_f32_r) && (compute_type == rocblas_datatype_f16_r))
    {
        DISPATCH(float, rocblas_half);
    }
    else if((type_C == rocblas_datatype_f32_r) && (compute_type == rocblas_datatype_bf16_r))
    {
        DISPATCH(float, rocblas_bfloat16);
    }
    else
    {
        // ------------------------------
        // type combination not supported
        // ------------------------------
        istat = rocblas_status_not_implemented;
    }
    return (istat);

#undef DISPATCH
#undef CHECK_MEM
#undef ROCBLAS_CHECK
}

ROCSOLVER_END_NAMESPACE
