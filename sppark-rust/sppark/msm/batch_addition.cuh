// 版权声明
// Apache 2.0 许可证，详情请参阅 LICENSE 文件
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_BATCH_ADDITION_CUH__
#define __SPPARK_MSM_BATCH_ADDITION_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <vector>

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#define BATCH_ADD_BLOCK_SIZE 256
#ifndef BATCH_ADD_NSTREAMS
# define BATCH_ADD_NSTREAMS 8
#elif BATCH_ADD_NSTREAMS == 0
# error "invalid BATCH_ADD_NSTREAMS"
#endif


template<class bucket_t, class affine_h,
        class bucket_h = class bucket_t::mem_t,
        class affine_t = class bucket_t::affine_t>
// 定义 batch_addition 函数模板，用于在 GPU 上执行向量批量加法运算
// bucket_t 为模板参数，表示加法的数据类型
// affine_h 表示输入的点集数据类型
// bucket_h 和 affine_t 分别表示 bucket_t 类型的主机和设备内存类型
__launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
void batch_addition(bucket_h ret[], const affine_h points[], uint32_t npoints,
                    const uint32_t bitmap[], bool accumulate = false,
                    uint32_t sid = 0)
{
    // 静态变量，保存每个流的计数器
    static __device__ uint32_t streams[BATCH_ADD_NSTREAMS];
    uint32_t& current = streams[sid % BATCH_ADD_NSTREAMS];

    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;

    // 计算线程和块索引
    const uint32_t tid = (threadIdx.x + blockDim.x*blockIdx.x) / degree;
    const uint32_t xid = tid % warp_sz;

    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));

    bucket_t acc;
    acc.inf();

    // 如果 accumulate 为真，将结果保存到 acc 中
    if (accumulate && tid < gridDim.x*blockDim.x/WARP_SZ)
        acc = ret[tid];

    // 每个块中的线程依次处理点集
    uint32_t base = laneid == 0 ? atomicAdd(&current, 32*WARP_SZ) : 0;
    base = __shfl_sync(0xffffffff, base, 0);

    while (base < npoints) {
        // 计算当前块的大小，最大为 32*WARP_SZ 或者剩余的点数
        uint32_t chunk = min(32*WARP_SZ, npoints-base);

        // 从位图中获取当前线程应当处理的位信息
        uint32_t bits = bitmap[base/WARP_SZ + laneid];

        // 遍历点集的位图，根据位图信息选择要处理的点
        for (uint32_t word, off = 0xffffffff, j = 0, i = 0; i < chunk;) {
            // 每次循环开始时，都从 bits 中获取一个新的字（word）
            if (i % 32 == 0)
                word = __shfl_sync(0xffffffff, bits, i/32);

            // 检查当前点是否需要处理
            if (word & 1) {
                if (j++ == xid)  // 如果是当前线程需要处理的点
                    off = i;      // 记录该点的索引
            }
            word >>= 1;  // 继续处理下一个点

            if (++i == chunk || j == warp_sz) {
                // 当处理完一个字或者达到 warp_sz 个点时，进行处理
                if (off != 0xffffffff) {
                    affine_t p = points[base + off];  // 获取对应点的数据
                    if (degree == 2)
                        acc.uadd(p);  // 执行累加操作
                    else
                        acc.add(p);
                }
                j = 0;      // 重置局部计数器
                off = 0xffffffff;  // 重置当前点的索引
            }
        }

        // 使用原子操作更新 base 的值，获取下一块要处理的点的起始索引
        base = laneid == 0 ? atomicAdd(&current, 32*WARP_SZ) : 0;
        base = __shfl_sync(0xffffffff, base, 0);  // 同步线程，获取更新后的 base 值
    }


#ifdef __CUDA_ARCH__
    for (uint32_t off = 1; off < warp_sz;) {
        // 在 warp 内进行归约操作，将结果相加
        // 通过 shfl_down 函数实现
        // 每个线程将自己的数据向下传递给离自己 off 个位置的线程
        // 循环进行，每次 off 增加一倍，直到 off 大于等于 warp_sz
        bucket_t down = acc.shfl_down(off*degree);

        off <<= 1;
        // 每轮循环只有部分线程参与相加，这里保证只有 xid 是 off 的整数倍的线程参与相加
        if ((xid & (off-1)) == 0)
            acc.uadd(down); // 使用 uadd 函数将 down 加到 acc 中
                            // 注意：.add() 触发寄存器溢出...在 .shfl_down() 中
    }
#endif


    cooperative_groups::this_grid().sync();

    // 将结果保存到 ret 中
    if (xid == 0)
        ret[tid/warp_sz] = acc;

    // 重置计数器为 0
    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

// 另一个 batch_addition 函数模板，用于在 GPU 上执行向量批量加法运算
// 与上面的函数不同，这里使用 digits 数组来指定要处理的点的索引
template<class bucket_t, class affine_h,
        class bucket_h = class bucket_t::mem_t,
        class affine_t = class bucket_t::affine_t>
__launch_bounds__(BATCH_ADD_BLOCK_SIZE) __global__
void batch_addition(bucket_h ret[], const affine_h points[], size_t npoints,
                    const uint32_t digits[], const uint32_t& ndigits)
{
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;

    // 计算线程和块索引
    const uint32_t tid = (threadIdx.x + blockDim.x*blockIdx.x) / degree;
    const uint32_t xid = tid % warp_sz;

    bucket_t acc;
    acc.inf();

    // 遍历 digits 数组，并根据数字选择要处理的点
    for (size_t i = tid; i < ndigits; i += gridDim.x*blockDim.x/degree) {
        uint32_t digit = digits[i];
        affine_t p = points[digit & 0x7fffffff];
        if (degree == 2)
            acc.uadd(p, digit >> 31);
        else
            acc.add(p, digit >> 31);
    }

#ifdef __CUDA_ARCH__
    for (uint32_t off = 1; off < warp_sz;) {
        bucket_t down = acc.shfl_down(off*degree);

        off <<= 1;
        if ((xid & (off-1)) == 0)
            acc.uadd(down); // .add() 触发寄存器溢出...在 .shfl_down() 中
    }

    // 将结果保存到 ret 中
    if (xid == 0)
        ret[tid/warp_sz] = acc;
}

// 对输入数组进行求和操作
template<class bucket_t>
bucket_t sum_up(const bucket_t inp[], size_t n)
{
    bucket_t sum = inp[0];
    for (size_t i = 1; i < n; i++)
        sum.add(inp[i]);
    return sum;
}

// 对输入向量进行求和操作
template<class bucket_t>
bucket_t sum_up(const std::vector<bucket_t>& inp)
{   return sum_up(&inp[0], inp.size());   }
#endif