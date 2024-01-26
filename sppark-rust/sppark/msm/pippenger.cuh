// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_CUH__
#define __SPPARK_MSM_PIPPENGER_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cassert>

#include <util/vec2d_t.hpp>
#include <util/slice_t.hpp>

#include "sort.cuh"
#include "batch_addition.cuh"

#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

typedef std::chrono::high_resolution_clock Clock;

using namespace std;
/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */

#ifdef __CUDA_ARCH__
// Transposed scalar_t
template<class scalar_t>
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    // 获取元素
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }

    // 获取指定 laneid 的元素
    __device__ scalar_T& operator()(uint32_t laneid)
    {   return *reinterpret_cast<scalar_T*>(&val[0][laneid]);   }

    // 赋值运算符重载，将 scalar_t 对象赋值给当前对象
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

// 获取指定位置的 wval
template<class scalar_t>
__device__ __forceinline__
static uint32_t get_wval(const scalar_T<scalar_t>& scalar, uint32_t off,
                         uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1)
{
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i)
        ret |= (uint64_t)scalar[i+1] << 32;

    return ret >> (off%32);
}

// Booth 编码
__device__ __forceinline__
static uint32_t booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits)
{
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0-wval : wval;
}
#endif

template<class scalar_t>
__launch_bounds__(1024) __global__
void breakdown(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
               uint32_t nwins, uint32_t wbits, bool mont = true)
{
    // len 最大为 2^31，wbits 必须小于 32
    assert(len <= (1U<<31) && wbits < 32);

#ifdef __CUDA_ARCH__
    // 声明共享内存
    extern __shared__ scalar_T<scalar_t> xchange[];
    // 获取线程编号和块编号
    const uint32_t tid = threadIdx.x;
    const uint32_t tix = threadIdx.x + blockIdx.x*blockDim.x;

    // 计算一些常量
    const uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1;
    const uint32_t wmask = 0xffffffffU >> (31-wbits); // (1U << (wbits+1)) - 1;

    / 获取当前线程对应的 scalar 对象
    auto& scalar = xchange[tid/WARP_SZ](tid%WARP_SZ);

    #pragma unroll 1
    for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x*blockDim.x) {
        auto s = scalars[i];

#if 0
        s.from();

        // 如果要进行 Montgomery 运算，则将 s 转换为 Montgomery 格式
        if (!mont) s.to();
#else
        if (mont) s.from();
#endif

        // 清除最高位，并根据最高位的值对 s 进行条件取反
        uint32_t msb = s[top_i] >> ((scalar_t::nbits - 1) % 32);
        s.cneg(msb);
        msb <<= 31;

        // 将 s 赋值给 scalar
        scalar = s;

        #pragma unroll 1
        for (uint32_t bit0 = nwins*wbits - 1, win = nwins; --win;) {
            bit0 -= wbits;
            // 获取 scalar 中指定位置的 wval，并进行 Booth 编码
            uint32_t wval = get_wval(scalar, bit0, top_i);
            wval = booth_encode(wval, wmask, wbits);
            // 如果 wval 不为零，则对其进行异或操作，得到编码结果
            if (wval) wval ^= msb;
            // 将编码结果存储到 digits 数组中
            digits[win][i] = wval;
        }
        // 处理最后一个窗口
        uint32_t wval = s[0] << 1;
        wval = booth_encode(wval, wmask, wbits);
        if (wval) wval ^= msb;
        digits[0][i] = wval;
    }
#endif
}

#ifndef LARGE_L1_CODE_CACHE
# if __CUDA_ARCH__-0 >= 800 // 如果当前GPU的架构号大于或等于800，则开启L1代码缓存
#  define LARGE_L1_CODE_CACHE 1
#  define ACCUMULATE_NTHREADS 384 // 线程数设置为384
# else // 否则关闭L1代码缓存
#  define LARGE_L1_CODE_CACHE 0
#  define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 384 : 256)  // 根据桶的度数决定线程数（如果是1，则设置为384，否则设置为256）
# endif
#endif

#ifndef MSM_NTHREADS
# define MSM_NTHREADS 256 // 如果没有定义MSM_NTHREADS，则设置默认线程数为256
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS-1)) != 0 // 检查线程数是否小于32或者不是2的幂次方
# error "bad MSM_NTHREADS value" // 若不满足上述条件，则程序报错并输出提示信息
#endif
#ifndef MSM_NSTREAMS
# define MSM_NSTREAMS 8 // 如果没有定义MSM_NSTREAMS，则设置默认流数为8
#elif MSM_NSTREAMS<2 // 如果MSM_NSTREAMS小于2，则说明其无效，程序报错并输出提示信息
# error "invalid MSM_NSTREAMS"
#endif


template<class bucket_t,
         class affine_h,
         class bucket_h = class bucket_t::mem_t,
         class affine_t = class bucket_t::affine_t>
// - bucket_t: 桶的类型
// - affine_h: 仿射变换的类型
// - bucket_h: 桶的内存类型，默认为bucket_t::mem_t
// - affine_t: 仿射变换的类型，默认为bucket_t::affine_t
__launch_bounds__(ACCUMULATE_NTHREADS) __global__  // 设置函数在GPU上运行时的线程块大小
void accumulate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits,
                const vec2d_t<uint32_t> histogram, uint32_t sid = 0)
{
    // 函数的入口点，接收桶数组、窗口数量、窗口位数、点阵数组、digits数组、histogram数组以及一个可选的sid参数
    vec2d_t<bucket_h> buckets{buckets_, 1U<<--wbits}; // 将桶数组转换为二维数组

    const affine_h* points = points_; // 将points_赋值给points指针

    static __device__ uint32_t streams[MSM_NSTREAMS]; // 在设备端定义一个静态数组，存储流号
    uint32_t& current = streams[sid % MSM_NSTREAMS]; // 根据sid获取当前流号
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid)); // 使用PTX汇编语句获取当前线程的laneid
    const uint32_t degree = bucket_t::degree; // 获取桶的度数
    const uint32_t warp_sz = WARP_SZ / degree; // 计算每个warp中的线程数
    const uint32_t lane_id = laneid / degree; // 计算lane的id

    uint32_t x, y;
    // 代码用于确定当前线程的x坐标和y坐标，用于索引到对应的桶和点阵元素。具体的逻辑根据条件编译进行了两种实现方式的选择。
#if 1 // 如果宏定义为1，使用共享内存的方式实现
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(&current, blockDim.x/degree); // 原子地将blockDim.x/degree加入到current中，并返回增加后的值
    __syncthreads();               // 等待所有线程执行完毕
    x = xchg + threadIdx.x/degree; // 计算当前线程的x坐标
#else // 如果宏定义为0，使用lane id的方式实现
    x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0; // 如果当前是warp中的第一个线程，则原子地将warp_sz加入到current中，并返回增加后的值；否则设置x为0
    x = __shfl_sync(0xffffffff, x, 0) + lane_id;        // 将x的值广播到warp中的所有线程，并加上当前线程的lane id
#endif

    while (x < (nwins << wbits)) {
        y = x >> wbits;         // 计算y坐标
        x &= (1U << wbits) - 1; // 计算x坐标，将x值截取到wbits位
        const uint32_t* h = &histogram[y][x]; // 获取对应的直方图元素

        uint32_t idx, len = h[0]; // 获取直方图元素中的索引和长度

        // 使用汇编指令进行条件选择和赋值操作，将len值广播到warp中的所有线程
        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}" : "=r"(idx) : "r"(len), "r"(degree));

        if (lane_id == 0 && x != 0)
            idx = h[-1]; // 如果当前是warp中的第一个线程，并且x不为零，则使用h[-1]的值更新idx

        if ((len -= idx) && !(x == 0 && y == 0)) { // 判断len是否大于idx且(x,y)是否为原点
            const uint32_t* digs_ptr = &digits[y][idx]; // 获取对应的digits数组的指针，并根据idx偏移
            uint32_t digit = *digs_ptr++; // 获取digits数组中的值，并将指针向后移动

            affine_t p = points[digit & 0x7fffffff]; // 使用digit的低31位作为索引获取points数组中的值
            bucket_t bucket = p; // 将p赋值给bucket

            bucket.cneg(digit >> 31); // 根据digit的最高位进行条件选择，并调用cneg函数进行操作

            while (--len) { // 循环处理剩余的len-1个元素
                digit = *digs_ptr++; // 获取digits数组中的值，并将指针向后移动
                p = points[digit & 0x7fffffff]; // 使用digit的低31位作为索引获取points数组中的值

                if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
                    bucket.add(p, digit >> 31); // 根据digit的最高位进行条件选择，并调用add函数进行操作
                else
                    bucket.uadd(p, digit >> 31); // 根据digit的最高位进行条件选择，并调用uadd函数进行操作
            }

            buckets[y][x] = bucket; // 将bucket赋值给对应的buckets数组元素
        } else {
            buckets[y][x].inf(); // 如果len不大于idx或者(x,y)为原点，则调用inf函数对buckets数组元素进行初始化
        }

        x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0; // 如果当前是warp中的第一个线程，则原子地将warp_sz加入到current中，并返回增加后的值；否则设置x为0
        x = __shfl_sync(0xffffffff, x, 0) + lane_id; // 将x的值广播到warp中的所有线程，并加上当前线程的lane id
    }

    cooperative_groups::this_grid().sync(); // 同步整个block内的所有线程

    if (threadIdx.x + blockIdx.x == 0)
        current = 0; // 如果当前是block中的第一个线程，则将current置为0
}

template<class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__
void integrate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits, uint32_t nbits)
// 函数接收四个参数，分别为存储多项式系数的数组指针、多项式的次数加1、每一项多项式系数所占据的位数以及整个多项式的位数。
// buckets_ 表示存储多项式系数的数组指针，
// nwins 表示多项式的次数加 1，
// wbits 表示每一项多项式系数所占据的位数，
// nbits 表示整个多项式的位数。
{
    // 定义常量degree，表示bucket_t结构体中多项式系数个数
    const uint32_t degree = bucket_t::degree;
    // 计算出每个线程需要处理的多项式系数个数Nthrbits
    uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

    // 对输入参数进行断言检查，保证blockDim.x是2的幂次方且wbits-1大于Nthrbits。
    assert((blockDim.x & (blockDim.x-1)) == 0 && wbits-1 > Nthrbits);

    // 定义二维数组buckets，存储多项式系数。其中，buckets_为存储多项式系数的数组指针，
    // 1U<<(wbits-1)表示多项式系数的个数
    vec2d_t<bucket_h> buckets{buckets_, 1U<<(wbits-1)};

    extern __shared__ uint4 scratch_[];
    // 使用共享内存scratch进行存储，scratch_为共享内存数组指针
    auto* scratch = reinterpret_cast<bucket_h*>(scratch_);

    // 获取当前线程在block中的id和block在grid中的id。
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t bid = blockIdx.x;

    // 根据tid和i定位到当前线程需要处理的多项式系数
    auto* row = &buckets[bid][0];
    uint32_t i = 1U << (wbits-1-Nthrbits);
    row += tid * i;

    // 根据block在grid中的id和wbits，计算出mask表示最后一个多项式系数的掩码，用于处理边界情况。
    uint32_t mask = 0;
    if ((bid+1)*wbits > nbits) {
        uint32_t lsbits = nbits - bid*wbits;
        mask = (1U << (wbits-lsbits)) - 1;
    }

    // 定义bucket_t类型的res和acc变量，表示存储当前值和前一个值。
    // 初始时将最后一个多项式系数赋值给acc
    bucket_t res, acc = row[--i];


    // 如果最后一个系数需要进行初始化
    // 根据mask的值判断是否需要对res进行初始化。如果需要，则调用res.inf()函数进行初始化；否则将最后一个多项式系数赋值给res
    if (i & mask) {
        if (sizeof(res) <= 128) res.inf();
        else                    scratch[tid].inf();
    } else {
        if (sizeof(res) <= 128) res = acc;
        else                    scratch[tid] = acc;
    }


    bucket_t p;

    #pragma unroll 1
    while (i--) {
        p = row[i];

        // 根据mask的值选择相应的操作，如果为1，则设置pc为2；否则设置为0
        uint32_t pc = i & mask ? 2 : 0;

    #pragma unroll 1
        // 通过循环从后向前遍历row数组中的多项式系数，并根据mask的值选择相应的操作。
        // 将acc和p相加得到新的多项式，并根据不同情况将结果存储在res或者scratch数组中。
        do {
            if (sizeof(bucket_t) <= 128) { // 如果sizeof(bucket_t)小于等于128，则调用p.add(acc)函数
                p.add(acc);
                if (pc == 1) {
                    res = p;
                } else {
                    acc = p;
                    if (pc == 0) p = res;
                }
            } else { // 如果sizeof(bucket_t)大于128，则调用p.uadd(acc)函数
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);

                if (pc == 1) {
                    scratch[tid] = p;
                } else {
                    acc = p;
                    if (pc == 0) p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }


    __syncthreads();

    // 存储p和acc到buckets数组中对应的位置上。
    buckets[bid][2*tid] = p;
    buckets[bid][2*tid+1] = acc;
}
// 取消对asm宏的定义
#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
// 实例化accumulate函数模板
template __global__
void accumulate<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[],
                                           uint32_t nwins, uint32_t wbits,
                                           /*const*/ affine_t::mem_t points_[],
                                           const vec2d_t<uint32_t> digits,
                                           const vec2d_t<uint32_t> histogram,
                                           uint32_t sid);
// 实例化batch_addition函数模板
template __global__
void batch_addition<bucket_t>(bucket_t::mem_t buckets[],
                              const affine_t::mem_t points[], size_t npoints,
                              const uint32_t digits[], const uint32_t& ndigits);
// 实例化integrate函数模板
template __global__
void integrate<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                         uint32_t wbits, uint32_t nbits);
// 实例化breakdown函数模板
template __global__
void breakdown<scalar_t>(vec2d_t<uint32_t> digits, const scalar_t scalars[],
                         size_t len, uint32_t nwins, uint32_t wbits, bool mont);
#endif

#include <vector>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

template<class bucket_t, class point_t, class affine_t, class scalar_t,
         class affine_h = class affine_t::mem_t,
         class bucket_h = class bucket_t::mem_t>
class msm_t {
    const gpu_t& gpu;  // GPU 实例的引用
    size_t npoints;  // 点的数量
    uint32_t wbits, nwins;  // 整数位数和窗口数量
    bucket_h *d_buckets;  // 存储桶指针
    affine_h *d_points;  // 点指针
    scalar_t *d_scalars;  // 标量指针
    vec2d_t<uint32_t> d_hist;  // 二维数组

    template<typename T> using vec_t = slice_t<T>;  // 模板别名

    // 结果类
    class result_t {
        bucket_t ret[MSM_NTHREADS/bucket_t::degree][2];  // 存储结果的数组
    public:
        result_t() {}  // 构造函数
        inline operator decltype(ret)&() { return ret; }  // 类型转换运算符
        inline const bucket_t* operator[](size_t i) const { return ret[i]; }  // 下标运算符
    };

    // 计算整数的对数
    constexpr static int lg2(size_t n) {
        int ret=0; while (n>>=1) ret++; return ret;
    }

public:
    // 构造函数，初始化各个参数
    msm_t(const affine_t points[], size_t np,
          size_t ffi_affine_sz = sizeof(affine_t), int device_id = -1)
            : gpu(select_gpu(device_id)), d_points(nullptr), d_scalars(nullptr)
    {
        // 对点的数量进行处理
        npoints = (np+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        // 设置整数位数和窗口数量
        wbits = 17;
        if (npoints > 192) {
            wbits = std::min(lg2(npoints + npoints/2) - 8, 18);
            if (wbits < 10)
                wbits = 10;
        } else if (npoints > 0) {
            wbits = 10;
        }
        nwins = (scalar_t::bit_length() - 1) / wbits + 1;

        // 计算数组大小
        uint32_t row_sz = 1U << (wbits-1);
        size_t d_buckets_sz = (nwins * row_sz)
                            + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        size_t d_blob_sz = (d_buckets_sz * sizeof(d_buckets[0]))
                         + (nwins * row_sz * sizeof(uint32_t))
                         + (points ? npoints * sizeof(d_points[0]) : 0);

        // 分配存储空间
        d_buckets = reinterpret_cast<decltype(d_buckets)>(gpu.Dmalloc(d_blob_sz));
        d_hist = vec2d_t<uint32_t>(&d_buckets[d_buckets_sz], row_sz);
        if (points) {
            d_points = reinterpret_cast<decltype(d_points)>(d_hist[nwins]);
            gpu.HtoD(d_points, points, np, ffi_affine_sz);
            npoints = np;
        } else {
            npoints = 0;
        }
    }

    // 其他构造函数的重载
    inline msm_t(vec_t<affine_t> points, size_t ffi_affine_sz = sizeof(affine_t),
                 int device_id = -1)
        : msm_t(points, points.size(), ffi_affine_sz, device_id) {};
    inline msm_t(int device_id = -1)
            : msm_t(nullptr, 0, 0, device_id) {};

    // 析构函数
    ~msm_t()
    {
        gpu.sync();
        if (d_buckets) gpu.Dfree(d_buckets);
    }

private:
    // 将标量数组中的每个元素进行数字分解，并将结果存储在二维数组中
    void digits(const scalar_t d_scalars[], size_t len,
                vec2d_t<uint32_t>& d_digits, vec2d_t<uint2>&d_temps, bool mont)
    {
        // Using larger grid size doesn't make 'sort' run faster, actually
        // quite contrary. Arguably because global memory bus gets
        // thrashed... Stepping far outside the sweet spot has significant
        // impact, 30-40% degradation was observed. It's assumed that all
        // GPUs are "balanced" in an approximately the same manner. The
        // coefficient was observed to deliver optimal performance on
        // Turing and Ampere...
        // 根据SM数量计算grid size，以确定CUDA kernel的线程块数量
        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));

        // 调用breakdown kernel进行分解操作，并传递相应参数
        breakdown<<<2*grid_size, 1024, sizeof(scalar_t)*1024, gpu[2]>>>(
            d_digits, d_scalars, len, nwins, wbits, mont
        );
        CUDA_OK(cudaGetLastError()); // 检查CUDA函数是否执行成功

        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS; // 计算shared memory的大小

#if 0
        uint32_t win;
        for (win = 0; win < nwins-1; win++) {
            // 调用sort kernel对数字进行排序
            gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, 0u);
        }
        uint32_t top = scalar_t::bit_length() - wbits * win;
        gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
#else
        // 并行地启动一对或一个sort kernel函数，以更高的并发性进行排序。
        // 如果剩余位数不为零，则最后一个kernel函数需要处理剩余的位数。
        // On the other hand a pair of kernels launched in parallel run
        // ~50% slower but sort twice as much data...
        uint32_t top = scalar_t::bit_length() - wbits * (nwins-1);
        uint32_t win;
        for (win = 0; win < nwins-1; win += 2) {
            gpu[2].launch_coop(sort, {{grid_size, 2}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, win == nwins-2 ? top-1 : wbits-1);
        }
        if (win < nwins) {
            gpu[2].launch_coop(sort, {{grid_size, 1}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
        }
#endif
    }

// 这段代码是一个公共函数，用于调用一个加密算法的核心计算过程。
// 该函数接受一系列参数，包括输出点 out、输入点 points、标量 scalars 等等。
// 函数将输入点和标量进行一系列计算，并将结果存储在输出点中。
public:
    RustError invoke(point_t& out, const affine_t* points_, size_t npoints,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
//        auto start = Clock::now();
        // 检查输入参数
        assert(this->npoints == 0 || npoints <= this->npoints);

        uint32_t lg_npoints = lg2(npoints + npoints/2);
        size_t batch = 1 << (std::max(lg_npoints, wbits) - wbits);
        batch >>= 6;
        batch = batch ? batch : 1;
        uint32_t stride = (npoints + batch - 1) / batch;    // 计算步长
        stride = (stride+WARP_SZ-1) & ((size_t)0-WARP_SZ);  // 调整步长为 WARP_SZ 的倍数

        // 创建存储计算结果的容器
        std::vector<result_t> res(nwins);  // 存储计算结果的容器
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);  // 存储中间结果的容器

        // 初始化输出点
        out.inf();  // 将输出点初始化为无穷远点
        point_t p;

        try {
            // |scalars| being nullptr means the scalars are pre-loaded to
            // |d_scalars|, otherwise allocate stride.
            // 分配设备内存用于临时存储计算过程中的数据
            size_t temp_sz = scalars ? sizeof(scalar_t) : 0;  // 计算标量所占内存大小
            temp_sz = stride * std::max(2*sizeof(uint2), temp_sz);  // 临时存储计算结果的大小

            // |points| being nullptr means the points are pre-loaded to
            // |d_points|, otherwise allocate double-stride.
            const char* points = reinterpret_cast<const char*>(points_);  // 将输入点转换为字符类型指针
            size_t d_point_sz = points ? (batch > 1 ? 2*stride : stride) : 0;  // 计算输入点所占内存大小
            d_point_sz *= sizeof(affine_h);  // 调整内存大小为 affine_h 数据类型的大小

            size_t digits_sz = nwins * stride * sizeof(uint32_t);  // 计算存储数字的内存大小

            dev_ptr_t<uint8_t> d_temp{temp_sz + digits_sz + d_point_sz, gpu[2]};  // 分配设备内存

            vec2d_t<uint2> d_temps{&d_temp[0], stride};  // 存储临时结果的二维数组
            vec2d_t<uint32_t> d_digits{&d_temp[temp_sz], stride};  // 存储数字的二维数组

            scalar_t* d_scalars = scalars ? (scalar_t*)&d_temp[0]
                                          : this->d_scalars;  // 设备上的标量数据指针
            affine_h* d_points = points ? (affine_h*)&d_temp[temp_sz + digits_sz]
                                        : this->d_points;  // 设备上的输入点数据指针

            // 初始化设备和主机的偏移量和数量
            size_t d_off = 0;   // 设备偏移量
            size_t h_off = 0;   // 主机偏移量
            size_t num = stride > npoints ? npoints : stride;  // 计算每次迭代的数量
            event_t ev;

            // 将输入标量从主机内存复制到设备内存
            if (scalars)
                gpu[2].HtoD(&d_scalars[d_off], &scalars[h_off], num);
            // 对标量进行一系列计算
            digits(&d_scalars[0], num, d_digits, d_temps, mont);
            gpu[2].record(ev);

            // 将输入点从主机内存复制到设备内存
            if (points)
                gpu[0].HtoD(&d_points[d_off], &points[h_off],
                            num,              ffi_affine_sz);

            // 循环进行一系列计算步骤
            for (uint32_t i = 0; i < batch; i++) {
                // 等待前一步计算完成
                gpu[i&1].wait(ev);

                // 进行批量加法运算
                batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE, 0, gpu[i&1]>>>(
                        &d_buckets[nwins << (wbits-1)], &d_points[d_off], num, &d_digits[0][0], d_hist[0][0]
                );
                CUDA_OK(cudaGetLastError());

                // 执行累积计算
                gpu[i&1].launch_coop(accumulate<bucket_t, affine_h>,
                    {gpu.sm_count(), 0},
                    d_buckets, nwins, wbits, &d_points[d_off], d_digits, d_hist, i&1
                );
                gpu[i&1].record(ev);

                // 执行积分计算
                integrate<bucket_t><<<nwins, MSM_NTHREADS,
                                      sizeof(bucket_t)*MSM_NTHREADS/bucket_t::degree,
                                      gpu[i&1]>>>(
                    d_buckets, nwins, wbits, scalar_t::bit_length()
                );
                CUDA_OK(cudaGetLastError());

                if (i < batch-1) {
                    h_off += stride;
                    num = h_off + stride <= npoints ? stride : npoints - h_off;

                    // 将标量数据从主机内存复制到设备内存
                    if (scalars)
                        gpu[2].HtoD(&d_scalars[0], &scalars[h_off], num);
                    gpu[2].wait(ev);

                    // 执行数字转换
                    digits(&d_scalars[scalars ? 0 : h_off], num, d_digits, d_temps, mont);
                    gpu[2].record(ev);

                    if (points) {
                        size_t j = (i + 1) & 1;
                        d_off = j ? stride : 0;

                        // 将点数据从主机内存复制到设备内存
                        gpu[j].HtoD(&d_points[d_off], &points[h_off*ffi_affine_sz], num, ffi_affine_sz);
                    } else {
                        d_off = h_off;
                    }
                }

                if (i > 0) {
                    // 收集结果
                    collect(p, res, ones);
                    out.add(p);
                }

                // 将最高位桶和结果从设备内存复制到主机内存
                gpu[i&1].DtoH(ones, d_buckets + (nwins << (wbits-1)));
                gpu[i&1].DtoH(res, d_buckets, sizeof(bucket_h)<<(wbits-1));
                gpu[i&1].sync();
            }
        } catch (const cuda_error& e) {
            gpu.sync();
            // 返回 CUDA 错误信息
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        collect(p, res, ones); //函数的作用是将累积后的结果和最高位桶的数据进行收集，得到最终的运算结果。其中，p表示输出结果的位置，res表示累积后的结果，ones表示最高位桶的数据。
        out.add(p); //是将运算得到的结果添加到输出中。out是一个输出对象，可以将结果保存到内存或磁盘等位置。

//        auto end = Clock::now();
//        uint64_t dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//        printf("MSM size %ld took %ld us\n", npoints, dt);

        return RustError{cudaSuccess};
    }

// RustError类型的invoke函数
// 将gpu_ptr_t<scalar_t>类型的scalars赋值给d_scalars，并调用含有更多参数的invoke函数
    RustError invoke(point_t& out, const affine_t* points, size_t npoints,
                                   gpu_ptr_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        d_scalars = scalars;
        return invoke(out, points, npoints, nullptr, mont, ffi_affine_sz);
    }

// RustError类型的invoke函数
// 调用含有更多参数的invoke函数，将vec_t<scalar_t>类型的scalars作为参数
    RustError invoke(point_t& out, vec_t<scalar_t> scalars, bool mont = true)
    {
        return invoke(out, nullptr, scalars.size(), scalars, mont);
    }

// RustError类型的invoke函数
// 调用含有更多参数的invoke函数，将vec_t<affine_t>类型的points和const scalar_t*类型的scalars作为参数
    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

// RustError类型的invoke函数
// 调用含有更多参数的invoke函数，将vec_t<affine_t>类型的points和vec_t<scalar_t>类型的scalars作为参数
    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   vec_t<scalar_t> scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);   }

// RustError类型的invoke函数
// 调用含有更多参数的invoke函数，将std::vector<affine_t>类型的points和std::vector<scalar_t>类型的scalars作为参数
    RustError invoke(point_t& out, const std::vector<affine_t>& points,
                                   const std::vector<scalar_t>& scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        // 调用含有更多参数的invoke函数，传入points.data()、points.size()和scalars.data()作为参数
        return invoke(out, points.data(),
                           std::min(points.size(), scalars.size()),
                           scalars.data(), mont, ffi_affine_sz);
    }

private:
    // 对每一行进行积分运算，返回积分结果
    point_t integrate_row(const result_t& row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_NTHREADS/bucket_t::degree);

        assert(wbits-1 > NTHRBITS);

        size_t i = MSM_NTHREADS/bucket_t::degree - 1;

        if (lsbits-1 <= NTHRBITS) {
            // 计算掩码
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)   res.inf();   // 当掩码不为0时，设置结果为无穷
            else        res = acc;   // 当掩码为0时，结果等于acc

            // 从最后一行开始向前计算累加结果
            while (i--) {
                acc.add(row[i][1]);   // 累加acc
                // 根据掩码判断是否需要累加到结果中
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res;
        }

        point_t  res = row[i][0];
        bucket_t acc = row[i][1];

        // 从倒数第二行开始向前计算积分结果
        while (i--) {
            point_t raise = acc;
            // 将raise乘以2的次幂
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            // 将raise加到结果中
            res.add(raise);
            // 将row[i][0]加到结果中
            res.add(row[i][0]);
            if (i)
                acc.add(row[i][1]);   // 累加acc
        }

        return res;
    }

    // 收集各个结果并计算最终积分结果
    void collect(point_t& out, const std::vector<result_t>& res,
                               const std::vector<bucket_t>& ones)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);   // 网格，用于存储每个网格块的信息

        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = scalar_t::bit_length() - y*wbits;
        total++;

        // 初始化网格信息
        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        // 获取可用CPU线程数
        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);   // 对每一行进行积分运算
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);   // 计算ones的总和

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);   // 将每个网格块的积分结果加到最终结果中
                if (y == 0)
                    break;
                // 将结果乘以2的次幂
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        // 将ones的总和加到最终结果中
        out.add(one);
    }
};

template<class bucket_t, class point_t, class affine_t, class scalar_t> static
RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[], bool mont = true,
                                       size_t ffi_affine_sz = sizeof(affine_t))
{
    // 创建并初始化 MSM 实例
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        // 调用 MSM 的 invoke 方法进行 Pippenger 算法计算
        return msm.invoke(*out, slice_t<affine_t>{points, npoints},
                                scalars, mont, ffi_affine_sz);
    } catch (const cuda_error& e) {
        out->inf(); // 设置输出结果为无穷大
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()}; // 返回错误消息
#else
        return RustError{e.code()}; // 返回错误代码
#endif
    }
}
#endif
