// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_BATCH_ADDITION_CUH__
#define __SPPARK_MSM_BATCH_ADDITION_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <vector>

// 定义WARP大小，默认为32
#ifndef WARP_SZ
# define WARP_SZ 32
#endif

// 定义批量加法操作的每个块的大小，默认为256
#define BATCH_ADD_BLOCK_SIZE 256
// 定义批量加法操作使用的流的数量，默认为8
#ifndef BATCH_ADD_NSTREAMS
# define BATCH_ADD_NSTREAMS 8
#elif BATCH_ADD_NSTREAMS == 0
# error "invalid BATCH_ADD_NSTREAMS"
#endif

template<class bucket_t, class affine_h,
        class bucket_h = class bucket_t::mem_t,
        class affine_t = class bucket_t::affine_t>
// 定义函数add
__device__ __forceinline__
static void add(bucket_h ret[], const affine_h points[], uint32_t npoints,
                const uint32_t bitmap[], const uint32_t refmap[],
                bool accumulate, uint32_t sid)
{
    // 定义静态变量streams，用于后续的流缓冲池
    static __device__ uint32_t streams[BATCH_ADD_NSTREAMS];
    uint32_t& current = streams[sid % BATCH_ADD_NSTREAMS];

    // 获取bucket_t模板参数中定义的常量degree，以及默认WARP大小除以degree得到的值warp_sz
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    // 获取当前线程的ID和线程所在的块的ID
    const uint32_t tid = (threadIdx.x + blockDim.x*blockIdx.x) / degree;
    const uint32_t xid = tid % warp_sz;

    // 获取当前线程的lane ID
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));

    // 定义一个bucket_t类型的变量acc，并初始化为inf()
    bucket_t acc;
    acc.inf();

    // 如果accumulate为true，且当前线程所处理的数据在ret数组中，将ret[tid]的值赋给acc
    if (accumulate && tid < gridDim.x*blockDim.x/WARP_SZ)
        acc = ret[tid];

    // 计算每个线程处理的数据的起始位置base，并确保所有线程都使用相同的值
    uint32_t base = laneid == 0 ? atomicAdd(&current, 32*WARP_SZ) : 0;
    base = __shfl_sync(0xffffffff, base, 0);

    // 计算当前chunk的大小，确保不超过npoints-base
    uint32_t chunk = min(32*WARP_SZ, npoints-base);
    uint32_t bits, refs, word, sign = 0, off = 0xffffffff;

    // 循环处理每个点
    for (uint32_t i = 0, j = 0; base < npoints;) {
        // 如果i==0，获取bitmap和refmap的值
        if (i == 0) {
            bits = bitmap[base/WARP_SZ + laneid];
            refs = refmap ? refmap[base/WARP_SZ + laneid] : 0;

            bits ^= refs;
            refs &= bits;
        }

        // 处理当前chunk中的每个点
        for (; i < chunk && j < warp_sz; i++) {
            // 如果已处理32个点，则从bits中获取新的32位数据
            if (i%32 == 0)
                word = __shfl_sync(0xffffffff, bits, i/32);
            // 如果refmap存在且已处理32个点，则从refs中获取新的32位数据
            if (refmap && (i%32 == 0))
                sign = __shfl_sync(0xffffffff, refs, i/32);

            // 如果当前bit为1，则表示该点需要加入计算
            if (word & 1) {
                // 如果当前线程是第j个处理该点的线程，则保存该点的位置和符号到off中
                if (j++ == xid)
                    off = (base + i) | (sign << 31);
            }
            word >>= 1;
            sign >>= 1;
        }

        // 如果chunk中的点都已处理完，则更新base并计算新的chunk大小
        if (i == chunk) {
            base = laneid == 0 ? atomicAdd(&current, 32*WARP_SZ) : 0;
            base = __shfl_sync(0xffffffff, base, 0);
            chunk = min(32*WARP_SZ, npoints-base);
            i = 0;
        }

        // 如果所有点都已处理完，或者当前处理的点数已达到warp_sz，则将当前累加结果加到acc中
        if (base >= npoints || j == warp_sz) {
            if (off != 0xffffffff) {
                affine_t p = points[off & 0x7fffffff];
                if (degree == 2)
                    acc.uadd(p, off >> 31);
                else
                    acc.add(p, off >> 31);
                off = 0xffffffff;
            }
            j = 0;
        }
    }

#ifdef __CUDA_ARCH__
    for (uint32_t off = 1; off < warp_sz;) {
        bucket_t down = acc.shfl_down(off*degree);

        off <<= 1;
        if ((xid & (off-1)) == 0)
            acc.uadd(down); // .add() triggers spills ... in .shfl_down()
    }
#endif

    // 使用协作线程组同步所有线程
    cooperative_groups::this_grid().sync();

    // 每个线程将acc中的结果保存到ret数组对应位置中
    if (xid == 0)
        ret[tid/warp_sz] = acc;

    // 如果当前线程是第一个块的第一个线程，则清空流缓冲池
    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

template<class bucket_t, class affine_h,
        class bucket_h = class bucket_t::mem_t,
        class affine_t = class bucket_t::affine_t>
//这是一个模板函数 batch_addition，用于进行批量加法操作。函数使用了 CUDA 的并行计算技术，以提高计算效率。具体功能如下：
//函数接受四个类型参数：bucket_t、affine_h、bucket_h 和 affine_t。其中 bucket_t 表示存储点坐标的数据类型，affine_h 和 affine_t 分别表示主机和设备上存储点坐标的数据类型，bucket_h 则表示主机上存储结果的数据类型。
//函数定义了一个静态变量 streams，用于后续的流缓冲池。
//函数获取当前线程的 ID 和线程所在的块的 ID，然后计算每个线程处理的数据的起始位置 base。
//函数处理每个点，将需要加入计算的点保存到 acc 中。
//在协作线程组同步后，每个线程将 acc 中的结果保存到 ret 数组对应位置中。
__launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
void batch_addition(bucket_h ret[], const affine_h points[], uint32_t npoints,
                    const uint32_t bitmap[], bool accumulate = false,
                    uint32_t sid = 0)
{   add<bucket_t>(ret, points, npoints, bitmap, nullptr, accumulate, sid);   }

//这是另一个模板函数 batch_diff，用于进行批量差分操作。函数的功能与 batch_addition 类似，不同之处在于它接受了额外的参数 refmap，用于指定参考映射。
template<class bucket_t, class affine_h,
        class bucket_h = class bucket_t::mem_t,
        class affine_t = class bucket_t::affine_t>
__launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
void batch_diff(bucket_h ret[], const affine_h points[], uint32_t npoints,
                const uint32_t bitmap[], const uint32_t refmap[],
                bool accumulate = false, uint32_t sid = 0)
{   add<bucket_t>(ret, points, npoints, bitmap, refmap, accumulate, sid);   }

// 这是另一个模板函数 batch_addition 的重载版本，用于计算一组数字对应的点的和。
// 函数首先计算每个数字对应的点的和，然后使用协作线程组技术将各个线程计算的结果合并到一个大小为 WARP_SZ 的 bucket_t 类型的数组中。
// 函数中的主要步骤如下：
// 根据桶的度数和每个 warp 的线程数计算当前线程的 ID 和 warp 内的 ID。
// 创建一个 bucket_t 类型的变量 acc，用于保存中间计算结果，并将其初始化为无穷大。
// 遍历每个数字，获取对应的点坐标，并根据桶的度数执行加法操作（degree 为 2 时为无符号加法）。
template<class bucket_t, class affine_h,
        class bucket_h = class bucket_t::mem_t,
        class affine_t = class bucket_t::affine_t>
__launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
void batch_addition(bucket_h ret[], const affine_h points[], size_t npoints,
                    const uint32_t digits[], const uint32_t& ndigits)
{
    const uint32_t degree = bucket_t::degree;  // 桶的度数
    const uint32_t warp_sz = WARP_SZ / degree;  // 每个warp的线程数
    const uint32_t tid = (threadIdx.x + blockDim.x * blockIdx.x) / degree;  // 当前线程ID
    const uint32_t xid = tid % warp_sz;  // 当前线程在warp内的ID

    bucket_t acc;  // 用于保存中间计算结果的变量
    acc.inf();  // 初始化acc为无穷大

    for (size_t i = tid; i < ndigits; i += gridDim.x * blockDim.x / degree) {
        uint32_t digit = digits[i];  // 获取当前处理的数字
        affine_t p = points[digit & 0x7fffffff];  // 获取对应的点坐标
        if (degree == 2)
            acc.uadd(p, digit >> 31);  // 执行加法操作（degree为2时为无符号加法）
        else
            acc.add(p, digit >> 31);  // 执行加法操作
    }

#ifdef __CUDA_ARCH__
    for (uint32_t off = 1; off < warp_sz;) {
        bucket_t down = acc.shfl_down(off*degree);

        off <<= 1;
        if ((xid & (off-1)) == 0)
            acc.uadd(down); // .add() triggers spills ... in .shfl_down()
    }
#endif

    if (xid == 0)
        ret[tid/warp_sz] = acc;
}

//这是一个非模板函数 sum_up，用于将一个包含多个 bucket_t 类型元素的数组中的元素相加得到一个 bucket_t 类型的结果。
//函数首先创建一个 bucket_t 类型的变量 ret，并将其初始化为无穷大。
//然后遍历数组中的每个元素，使用 add 方法将其累加到 ret 中。最后返回得到的结果 ret。
template<class bucket_t>
bucket_t sum_up(const bucket_t inp[], size_t n)
{
    // 创建一个变量 sum，并初始化为 inp[0]
    bucket_t sum = inp[0];

    // 遍历 inp 数组中的元素，从第二个元素开始循环
    for (size_t i = 1; i < n; i++)
        // 将当前元素累加到 sum 中
        sum.add(inp[i]);

    // 返回累加的结果 sum
    return sum;
}

//第二个函数 sum_up 接受一个向量 inp，并通过调用第一个函数 sum_up 来实现相同的功能。
//它使用 &inp[0] 获取向量的首地址，并使用 inp.size() 获取向量的大小作为参数传递给第一个函数。
template<class bucket_t>
bucket_t sum_up(const std::vector<bucket_t>& inp)
{   // 调用上面的函数 sum_up，并传入 inp 数组的首地址和大小
    return sum_up(&inp[0], inp.size());
}
#endif