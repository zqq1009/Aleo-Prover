// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_NTT_NTT_CUH__
#define __SPPARK_NTT_NTT_CUH__

#include <cassert>
#include <iostream>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include "parameters.cuh"
#include "kernels.cu"

#ifndef __CUDA_ARCH__

class NTT {
public:
    enum class InputOutputOrder { NN, NR, RN, RR };
    enum class Direction { forward, inverse };
    enum class Type { standard, coset };
    enum class Algorithm { GS, CT };

protected:
    static void bit_rev(fr_t* d_out, const fr_t* d_inp,
                        uint32_t lg_domain_size, stream_t& stream)
    {
        assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

        size_t domain_size = (size_t)1 << lg_domain_size;
        // aim to read 4 cache lines of consecutive data per read
        const uint32_t Z_COUNT = 256 / sizeof(fr_t);
        const uint32_t bsize = Z_COUNT>WARP_SZ ? Z_COUNT : WARP_SZ;

        if (domain_size <= 1024)
            bit_rev_permutation<<<1, domain_size, 0, stream>>>
                               (d_out, d_inp, lg_domain_size);
        else if (domain_size < bsize * Z_COUNT)
            bit_rev_permutation<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                               (d_out, d_inp, lg_domain_size);
        else if (Z_COUNT > WARP_SZ || lg_domain_size <= 32)
            bit_rev_permutation_z<<<domain_size / Z_COUNT / bsize, bsize,
                                    bsize * Z_COUNT * sizeof(fr_t), stream>>>
                                 (d_out, d_inp, lg_domain_size);
        else
            // Those GPUs that can reserve 96KB of shared memory can
            // schedule 2 blocks to each SM...
            bit_rev_permutation_z<<<gpu_props(stream).multiProcessorCount*2, 192,
                                    192 * Z_COUNT * sizeof(fr_t), stream>>>
                                 (d_out, d_inp, lg_domain_size);

        CUDA_OK(cudaGetLastError());
    }

private:
    static void LDE_powers(fr_t* inout, bool innt, bool bitrev,
                           uint32_t lg_dsz, uint32_t lg_blowup,
                           stream_t& stream)
    {
        size_t domain_size = (size_t)1 << lg_dsz;
        const auto gen_powers =
            NTTParameters::all(innt)[stream]->partial_group_gen_powers;

        if (domain_size < WARP_SZ)
            LDE_distribute_powers<<<1, domain_size, 0, stream>>>
                                 (inout, lg_dsz, lg_blowup, bitrev, gen_powers);
        else if (lg_dsz < 32)
            LDE_distribute_powers<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                                 (inout, lg_dsz, lg_blowup, bitrev, gen_powers);
        else
            LDE_distribute_powers<<<gpu_props(stream).multiProcessorCount, 1024,
                                    0, stream>>>
                                 (inout, lg_dsz, lg_blowup, bitrev, gen_powers);

        CUDA_OK(cudaGetLastError());
    }

protected:
    static void NTT_internal(fr_t* d_inout, uint32_t lg_domain_size,
                             InputOutputOrder order, Direction direction,
                             Type type, stream_t& stream)
    {
        // Pick an NTT algorithm based on the input order and the desired output
        // order of the data. In certain cases, bit reversal can be avoided which
        // results in a considerable performance gain.

        const bool intt = direction == Direction::inverse;
        const auto& ntt_parameters = *NTTParameters::all(intt)[stream];
        bool bitrev;
        Algorithm algorithm;

        switch (order) {
            case InputOutputOrder::NN:
                bit_rev(d_inout, d_inout, lg_domain_size, stream);
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::NR:
                bitrev = false;
                algorithm = Algorithm::GS;
                break;
            case InputOutputOrder::RN:
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::RR:
                bitrev = true;
                algorithm = Algorithm::GS;
                break;
            default:
                assert(false);
        }

        if (!intt && type == Type::coset)
            LDE_powers(d_inout, intt, bitrev, lg_domain_size, 0, stream);

        switch (algorithm) {
            case Algorithm::GS:
                GS_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
            case Algorithm::CT:
                CT_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
        }

        if (intt && type == Type::coset)
            LDE_powers(d_inout, intt, !bitrev, lg_domain_size, 0, stream);

        if (order == InputOutputOrder::RR)
            bit_rev(d_inout, d_inout, lg_domain_size, stream);
    }

public:
    static RustError Base(const gpu_t& gpu, fr_t* inout, uint32_t lg_domain_size,
                          InputOutputOrder order, Direction direction,
                          Type type)
    {
        // 如果域大小为0，则直接返回成功状态
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            // 选择指定的 GPU 设备
            gpu.select();

            // 计算域的实际大小
            size_t domain_size = (size_t)1 << lg_domain_size;
            // 在 GPU 上分配内存并将输入数据复制到 GPU 内存中
            dev_ptr_t<fr_t> d_inout{domain_size, gpu};
            gpu.HtoD(&d_inout[0], inout, domain_size);

            // 执行 NTT_internal 函数，在 GPU 上执行 NTT 操作
            NTT_internal(&d_inout[0], lg_domain_size, order, direction, type, gpu);

            // 将结果从 GPU 内存中复制回主机内存
            gpu.DtoH(inout, &d_inout[0], domain_size);
            // 同步 GPU，确保所有操作完成
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

protected:
    static void LDE_launch(stream_t& stream,
                           fr_t* ext_domain_data, fr_t* domain_data,
                           const fr_t (*gen_powers)[WINDOW_SIZE],
                           uint32_t lg_domain_size, uint32_t lg_blowup,
                           bool perform_shift = true, bool ext_pow = false)
    {
        assert(lg_domain_size + lg_blowup <= MAX_LG_DOMAIN_SIZE);
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size << lg_blowup;

        const cudaDeviceProp& gpu_prop = gpu_props(stream.id());

        // Determine the max power of 2 SM count
        size_t kernel_sms = gpu_prop.multiProcessorCount;
        while (kernel_sms & (kernel_sms - 1))
            kernel_sms -= (kernel_sms & (0 - kernel_sms));

        size_t device_max_threads = kernel_sms * 1024;
        uint32_t num_blocks, block_size;

        if (device_max_threads < domain_size) {
            num_blocks = kernel_sms;
            block_size = 1024;
        } else if (domain_size < 1024) {
            num_blocks = 1;
            block_size = domain_size;
        } else {
            num_blocks = domain_size / 1024;
            block_size = 1024;
        }

        stream.launch_coop(LDE_spread_distribute_powers,
                        {dim3(num_blocks), dim3(block_size),
                         sizeof(fr_t) * block_size},
                        ext_domain_data, domain_data, gen_powers,
                        lg_domain_size, lg_blowup, perform_shift, ext_pow);
    }

public:
    static RustError LDE_aux(const gpu_t& gpu, fr_t* inout,
                             uint32_t lg_domain_size, uint32_t lg_blowup,
                             fr_t *aux_out = nullptr)
    {
        try {
            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size << lg_blowup;
            size_t aux_size = aux_out != nullptr ? domain_size : 0;
            // The 2nd to last 'domain_size' chunk will hold the original data
            // The last chunk will get the bit reversed iNTT data
            dev_ptr_t<fr_t> d_inout{ext_domain_size + aux_size, gpu}; // + domain_size for aux buffer
            fr_t* aux_data = &d_inout[ext_domain_size];
            fr_t* domain_data = &d_inout[ext_domain_size - domain_size]; // aligned to the end
            fr_t* ext_domain_data = &d_inout[0];

            gpu.HtoD(domain_data, inout, domain_size);

            NTT_internal(domain_data, lg_domain_size,
                         InputOutputOrder::NR, Direction::inverse,
                         Type::standard, gpu);

            const auto gen_powers =
                NTTParameters::all()[gpu.id()]->partial_group_gen_powers;

            event_t sync_event;

            if (aux_out != nullptr) {
                bit_rev(aux_data, domain_data, lg_domain_size, gpu);
                sync_event.record(gpu);
            }

            LDE_launch(gpu, ext_domain_data, domain_data, gen_powers,
                       lg_domain_size, lg_blowup);

            // NTT - RN
            NTT_internal(ext_domain_data, lg_domain_size + lg_blowup,
                         InputOutputOrder::RN, Direction::forward,
                         Type::standard, gpu);

            if (aux_out != nullptr) {
                sync_event.wait(gpu[0]);
                gpu[0].DtoH(aux_out, aux_data, aux_size);
            }
            gpu.DtoH(inout, ext_domain_data, ext_domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    static RustError LDE(const gpu_t& gpu, fr_t* inout,
                         uint32_t lg_domain_size, uint32_t lg_blowup)
    {   return LDE_aux(gpu, inout, lg_domain_size, lg_blowup);   }

    static void Base_dev_ptr(stream_t& stream, fr_t* d_inout,
                             uint32_t lg_domain_size, InputOutputOrder order,
                             Direction direction, Type type)
    {
        size_t domain_size = (size_t)1 << lg_domain_size;

        NTT_internal(&d_inout[0], lg_domain_size, order, direction, type,
                     stream);
    }

    static void LDE_powers(stream_t& stream, fr_t* d_inout,
                           uint32_t lg_domain_size)
    {
        LDE_powers(d_inout, false, true, lg_domain_size, 0, stream);
    }

    // If d_out and d_in overlap, d_out is expected to encompass d_in and
    // d_in is expected to be aligned to the end of d_out
    // The input is expected to be in bit-reversed order
    static void LDE_expand(stream_t& stream, fr_t* d_out, fr_t* d_in,
                           uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        LDE_launch(stream, d_out, d_in, NULL, lg_domain_size, lg_blowup, false);
    }
};

#endif
#endif
