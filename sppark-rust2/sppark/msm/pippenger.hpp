// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_HPP__
#define __SPPARK_MSM_PIPPENGER_HPP__

#include <vector>
#include <memory>
#include <tuple>

/* Works up to 25 bits. */
static size_t get_wval(const unsigned char *d, size_t off, size_t bits) {
    // 计算位于字节数组中的起始位置和结束位置
    size_t i, top = (off + bits - 1) / 8;
    size_t ret, mask = (size_t)0 - 1;

    // 将指针移动到起始位置
    d += off / 8;
    // 计算要遍历的字节数
    top -= off / 8 - 1;

    /* this is not about constant-time-ness, but branch optimization */
    // 使用循环逐个字节读取数据，并根据条件进行控制
    for (ret = 0, i = 0; i < 4;) {
        // 将当前字节的值与掩码进行按位与操作，然后左移相应的位数
        ret |= (*d & mask) << (8 * i);
        // 更新掩码，用于判断是否需要跳过下一个字节
        mask = (size_t)0 - ((++i - top) >> (8 * sizeof(top) - 1));
        // 根据掩码判断是否需要移动指针到下一个字节
        d += 1 & mask;
    }

    // 返回结果，并根据起始位偏移进行右移操作
    return ret >> (off % 8);
}

static inline size_t window_size(size_t npoints)
{
    size_t wbits;

    // 计算滑动窗口的大小
    for (wbits = 0; npoints >>= 1; wbits++) ;

    // 根据计算结果返回具体的窗口大小
    return wbits > 12 ? wbits - 3 : (wbits > 4 ? wbits - 2 : (wbits ? 2 : 1));
}

template<class point_t, class bucket_t>
static void integrate_buckets(point_t& out, bucket_t buckets[], size_t wbits)
{
    bucket_t ret, acc;
    size_t n = (size_t)1 << wbits;

    /* Calculate sum of x[i-1]*i for i=1 through 1<<|wbits|. */
    // 计算 x[i-1]*i 的累加和，i 的取值范围是 1 到 1<<wbits
    acc = buckets[--n];
    ret = buckets[n];
    buckets[n].inf();
    while (n--) {
        acc.add(buckets[n]);
        ret.add(acc);
        buckets[n].inf();
    }
    out = ret;
}

template<class bucket_t, class affine_t>
static void bucket(bucket_t buckets[], size_t booth_idx,
                   size_t wbits, const affine_t& p)
{
    booth_idx &= (1<<wbits) - 1;
    if (booth_idx--)
        // 对指定索引的桶进行添加操作
        buckets[booth_idx].add(p);
}

// 预取函数:它的作用是根据指定的索引 booth_idx 预取 buckets 数组中的元素。预取是为了提前将数据从内存加载到 CPU 缓存中，以便后续的访问可以更快地进行。
template<class bucket_t>
static void prefetch(const bucket_t buckets[], size_t booth_idx, size_t wbits)
{
#if 0
    booth_idx &= (1<<wbits) - 1;
    if (booth_idx--)
        vec_prefetch(&buckets[booth_idx], sizeof(buckets[booth_idx]));
#else
    (void)buckets;      // 禁用预取，忽略 buckets 参数
    (void)booth_idx;    // 禁用预取，忽略 booth_idx 参数
    (void)wbits;        // 禁用预取，忽略 wbits 参数
#endif
}

template<class point_t, class affine_t, class bucket_t>
static void tile(point_t& ret, const affine_t points[], size_t npoints,
                 const unsigned char* scalars, size_t nbits,
                 bucket_t buckets[], size_t bit0, size_t wbits, size_t cbits)
{
    size_t wmask, wval, wnxt; // 定义变量，wmask为掩码，wval为当前值，wnxt为下一个值
    size_t i, nbytes; // 定义循环变量和字节数

    nbytes = (nbits + 7)/8; // 将比特位数转换为字节数
    wmask = ((size_t)1 << wbits) - 1; // 根据给定的比特位数计算掩码
    wval = get_wval(scalars, bit0, wbits) & wmask; // 获取第一个值，并与掩码进行按位与操作
    scalars += nbytes; // 指针指向下一个标量
    wnxt = get_wval(scalars, bit0, wbits) & wmask; // 获取下一个值，并与掩码进行按位与操作
    npoints--;  /* account for prefetch */ // 调整点数，减去预取的点数

    bucket(buckets, wval, cbits, points[0]); // 将第一个点添加到对应的桶中
    for (i = 1; i < npoints; i++) { // 遍历剩余的点
        wval = wnxt; // 更新当前值为下一个值
        scalars += nbytes; // 指针指向下一个标量
        wnxt = get_wval(scalars, bit0, wbits) & wmask; // 获取下一个值，并与掩码进行按位与操作
        prefetch(buckets, wnxt, cbits); // 对下一个桶进行预取
        bucket(buckets, wval, cbits, points[i]); // 将当前点添加到对应的桶中
    }
    bucket(buckets, wnxt, cbits, points[i]); // 将最后一个点添加到对应的桶中
    integrate_buckets(ret, buckets, cbits); // 整合桶的结果并存储到ret中
}

// 计算一个整数类型的二进制表示中的位数
template<typename T>
static size_t num_bits(T l)
{
    const size_t T_BITS = 8 * sizeof(T); // 定义整数类型的位数
# define MSB(x) ((T)(x) >> (T_BITS-1)) // 定义计算最高有效位的宏函数
    T x, mask; // 定义变量，x用于保存右移后的值，mask用于保存掩码

    if ((T)-1 < 0) {  // 处理有符号整数类型
        mask = MSB(l); // 获取最高有效位
        l ^= mask; // 取反
        l += 1 & mask; // 加上最高有效位的值
    }

    size_t bits = (((T)(~l & (l - 1)) >> (T_BITS - 1)) & 1) ^ 1; // 计算位数

    if (sizeof(T) > 4) {
        x = l >> (32 & (T_BITS - 1));   // 右移32位或者16位
        mask = MSB(0 - x);              // 获取最高有效位
        if ((T)-1 > 0) mask = 0 - mask; // 处理有符号整数类型
        bits += 32 & mask;      // 计算位数
        l ^= (x ^ l) & mask;    // 取反
    }

    if (sizeof(T) > 2) {
        x = l >> 16;        // 右移16位或者8位
        mask = MSB(0 - x);  // 获取最高有效位
        if ((T)-1 > 0) mask = 0 - mask; // 处理有符号整数类型
        bits += 16 & mask;   // 计算位数
        l ^= (x ^ l) & mask; // 取反
    }

    if (sizeof(T) > 1) {
        x = l >> 8; // 右移8位或者4位
        mask = MSB(0 - x); // 获取最高有效位
        if ((T)-1 > 0) mask = 0 - mask; // 处理有符号整数类型
        bits += 8 & mask; // 计算位数
        l ^= (x ^ l) & mask; // 取反
    }

    x = l >> 4; // 右移4位或者2位
    mask = MSB(0 - x); // 获取最高有效位
    if ((T)-1 > 0) mask = 0 - mask; // 处理有符号整数类型
    bits += 4 & mask; // 计算位数
    l ^= (x ^ l) & mask; // 取反

    x = l >> 2; // 右移2位或者1位
    mask = MSB(0 - x); // 获取最高有效位
    if ((T)-1 > 0) mask = 0 - mask; // 处理有符号整数类型
    bits += 2 & mask; // 计算位数
    l ^= (x ^ l) & mask; // 取反

    bits += l >> 1; // 计算位数

    return bits; // 返回位数
# undef MSB
}

//这段代码用于根据给定的位数（nbits）、窗口大小（window）和CPU数目（ncpus），计算出分解后的子任务数量。具体的逻辑如下：
//如果待处理的位数大于窗口大小乘以CPU数目，则将问题拆分为一个维度和一个窗口大小。初始化维度数量nx为1。
//如果窗口大小加上ncpus / 4所需的位数大于18，则将窗口大小减去ncpus / 4所需的位数；
// 否则，根据公式(nbits / window + ncpus - 1) / ncpus计算出初值窗口大小wnd，并比较(nbits / (window+1) + ncpus - 1) / ncpus与初值wnd的大小，选择较大值作为最终的窗口大小。
//如果待处理的位数不大于窗口大小乘以CPU数目，则将问题拆分为两个维度和一个窗口大小。初始化维度数量nx为2，窗口大小wnd为window - 2。
//利用循环，根据公式(nbits / wnd + 1) * nx < ncpus，逐步增加维度数量nx的值，并计算新的窗口大小。
//最后，根据给定的位数、窗口大小和计算得到的子任务数量ny，重新计算窗口大小wnd的值。
//返回一个包含维度数量nx、子任务数量ny和窗口大小wnd的std::tuple类型对象作为结果。
std::tuple<size_t, size_t, size_t>
static breakdown(size_t nbits, size_t window, size_t ncpus)
{
    size_t nx, ny, wnd; // 定义变量：nx表示维度数量，ny表示子任务数量，wnd表示窗口大小

    if (nbits > window * ncpus) { // 如果待处理的位数大于窗口大小乘以CPU数目
        nx = 1; // 维度数量为1

        if (window + (wnd = num_bits(ncpus / 4)) > 18) { // 如果窗口大小加上ncpus/4所需的位数大于18
            wnd = window - wnd; // 窗口大小减去ncpus/4所需的位数
        } else {
            wnd = (nbits / window + ncpus - 1) / ncpus; // 计算初值wnd为(nbits/window+ncpus-1)/ncpus
            if ((nbits / (window+1) + ncpus - 1) / ncpus < wnd) // 如果(nbits/(window+1)+ncpus-1)/ncpus小于初值wnd
                wnd = window + 1; // 窗口大小为window+1
            else
                wnd = window; // 窗口大小为window
        }
    } else { // 如果待处理的位数不大于窗口大小乘以CPU数目
        nx = 2; // 维度数量为2
        wnd = window - 2; // 窗口大小为window-2

        while ((nbits / wnd + 1) * nx < ncpus) { // 循环，直到(nx*窗口大小)大于等于CPU数目
            nx += 1; // 维度数量加1
            wnd = window - num_bits(3 * nx / 2); // 计算新的窗口大小
        }

        nx -= 1; // 维度数量减去1
        wnd = window - num_bits(3 * nx / 2); // 计算最终的窗口大小
    }

    ny = nbits / wnd + 1; // 子任务数量为nbits/wnd+1
    wnd = nbits / ny + 1; // 窗口大小重新计算为nbits/ny+1

    return std::make_tuple(nx, ny, wnd); // 返回包含nx、ny和wnd的tuple
}

// 函数模板：多次执行点的运算
template <class point_t, class affine_t, typename pow_t>
static void mult(point_t& ret, const affine_t& point,
                 const pow_t scalar, size_t top)
{
    ret.inf(); // 将返回值ret设为无穷远点
    if (point.is_inf()) // 如果输入的point是无穷远点，则直接返回
        return;

    struct is_bit { // 内部结构体：用于判断标量scalar的某一位是否为1
        static bool set(const pow_t v, size_t i)
        {   return (v[i/8] >> (i%8)) & 1;   } // 判断标量scalar的第i位是否为1
    };

    while (--top && !is_bit::set(scalar, top)) ; // 从高位开始逐位向低位遍历，找到第一个非零位

    if (is_bit::set(scalar, top)) { // 如果标量scalar的最高位为1
        ret = point; // 将点point赋值给返回值ret
        while (top--) { // 从标量scalar的第二高位开始，依次处理每一位
            ret.dbl(); // 对点ret进行倍乘运算
            if (is_bit::set(scalar, top)) // 如果当前位为1
                ret.add(point); // 将点point加到ret上
        }
    }
}


#include <util/thread_pool_t.hpp>

// 这段代码实现了 Pippenger 算法中的点倍乘运算。
//具体实现流程如下：
//1、首先根据标量位数和点数量计算出窗口大小以及分解窗口大小。
//2、利用分解窗口大小将任务平均分配给多个线程，并使用桶数据结构来存储点的信息。
//3、对于每个点，将其转换为仿射坐标并计算其切片值，即将点坐标按照窗口大小进行分组，并将每组中点的坐标相加。
//4、利用桶数据结构，将所有点的切片值加入到对应的桶中。
//5、从最高位开始，依次对每个窗口内的切片值进行加法运算，并将结果累加到输出点上。
//6、在每个窗口结束时，进行加倍运算。
//7、最后输出点即为所有点的倍数之和。
//8、总体来说，该算法的主要思想是对点进行切片，将其分组并按照窗口大小进行加法运算，最后进行加倍运算得到最终结果。使用桶数据结构可以避免重复计算，从而提高算法效率。同时，多线程的使用也可以加速运算过程。
template <class bucket_t, class point_t, class scalar_t,
          class affine_t = class bucket_t::affine_t>
static void mult_pippenger(point_t& ret, const affine_t points[], size_t npoints,
                           const scalar_t _scalars[], bool mont,
                           thread_pool_t* da_pool = nullptr)
{
    typedef typename scalar_t::pow_t pow_t;
    size_t nbits = scalar_t::nbits; // 标量位数
    size_t window = window_size(npoints); // 窗口大小
    size_t ncpus = da_pool ? da_pool->size() : 0; // 使用的线程数

    // 小端依赖，是否应该移除？
    const pow_t* scalars = reinterpret_cast<decltype(scalars)>(_scalars); // 标量数组
    std::unique_ptr<pow_t[]> store = nullptr;
    if (mont) {
        store = decltype(store)(new pow_t[npoints]);
        if (ncpus < 2 || npoints < 1024) {
            for (size_t i = 0; i < npoints; i++)
                _scalars[i].to_scalar(store[i]); // 转换标量为实际的数值
        } else {
            da_pool->par_map(npoints, 512, [&](size_t i) {
                _scalars[i].to_scalar(store[i]);
            });
        }
        scalars = &store[0];
    }

    if (ncpus < 2 || npoints < 32) {
        if (npoints == 1) { // 如果只有一个点
            mult(ret, points[0], scalars[0], nbits); // 直接进行乘法运算
            return;
        }

        std::vector<bucket_t> buckets(1 << window); /* zeroed */ // 桶数组

        point_t p;
        ret.inf(); // 初始化输出点为无穷远点

        /* top excess bits modulo target window size */
        size_t wbits = nbits % window, /* yes, it may be zero */
               cbits = wbits + 1,
               bit0 = nbits;
        while (bit0 -= wbits) {
            tile(p, points, npoints, scalars[0], nbits,
                 &buckets[0], bit0, wbits, cbits); // 进行切片运算
            ret.add(p); // 累加切片运算结果
            for (size_t i = 0; i < window; i++)
                ret.dbl(); // 进行加倍运算
            cbits = wbits = window;
        }
        tile(p, points, npoints, scalars[0], nbits,
                &buckets[0], 0, wbits, cbits);
        ret.add(p);
        return;
    }

    size_t nx, ny;
    std::tie(nx, ny, window) = breakdown(nbits, window, ncpus); // 计算分解窗口大小

    struct tile_t {
        size_t x, dx, y, dy;
        point_t p;
        tile_t() {}
    };
    std::vector<tile_t> grid(nx * ny);

    size_t dx = npoints / nx,
           y  = window * (ny - 1);

    size_t total = 0;
    while (total < nx) {
        grid[total].x  = total * dx;
        grid[total].dx = dx;
        grid[total].y  = y;
        grid[total].dy = nbits - y;
        total++;
    }
    grid[total - 1].dx = npoints - grid[total - 1].x;

    while (y) {
        y -= window;
        for (size_t i = 0; i < nx; i++, total++) {
            grid[total].x  = grid[i].x;
            grid[total].dx = grid[i].dx;
            grid[total].y  = y;
            grid[total].dy = window;
        }
    }

    std::vector<std::atomic<size_t>> row_sync(ny); /* zeroed */ // 行同步数组
    counter_t<size_t> counter(0); // 计数器
    channel_t<size_t> ch; // 通道

    auto n_workers = std::min(ncpus, total);
    while (n_workers--) {
        da_pool->spawn([&, window, total, nbits, nx, counter]() {
            size_t work;
            if ((work = counter++) < total) {
                std::vector<bucket_t> buckets(1 << window); /* zeroed */ // 桶数组

                do {
                    size_t x  = grid[work].x,
                           dx = grid[work].dx,
                           y  = grid[work].y,
                           dy = grid[work].dy;
                    tile(grid[work].p, &points[x], dx,
                                       scalars[x], nbits, &buckets[0],
                                       y, dy, dy + (dy < window));
                    if (++row_sync[y / window] == nx)
                        ch.send(y); // 发送信号通知行同步
                } while ((work = counter++) < total);
            }
        });
    }

    ret.inf();
    size_t row = 0;
    while (ny--) {
        auto y = ch.recv(); // 接收行同步信号
        row_sync[y / window] = -1U;
        while (grid[row].y == y) {
            while (row < total && grid[row].y == y)
                ret.add(grid[row++].p); // 累加切片运算结果
            if (y == 0)
                break;
            for (size_t i = 0; i < window; i++)
                ret.dbl(); // 进行加倍运算
            y -= window;
            if (row_sync[y / window] != -1U)
                break;
        }
    }
}

template <class bucket_t, class point_t, class scalar_t,
          class affine_t = class bucket_t::affine_t>
static void mult_pippenger(point_t& ret, const std::vector<affine_t>& points,
                           const std::vector<scalar_t>& scalars, bool mont,
                           thread_pool_t* da_pool = nullptr)
{
    // 点倍乘函数的封装，输入为点向量和标量向量，输出为倍数之和
    // 该函数使用了 std::vector 来存储点和标量，调用时比较方便

    // 调用底层的点倍乘函数，将点和标量转换为指针并传入函数中
    mult_pippenger<bucket_t>(ret, points.data(),
                                  std::min(points.size(), scalars.size()),
                                  scalars.data(), mont, da_pool);
}

#include <util/slice_t.hpp>

template <class bucket_t, class point_t, class scalar_t,
          class affine_t = class bucket_t::affine_t>
static void mult_pippenger(point_t& ret, slice_t<affine_t> points,
                           slice_t<scalar_t> scalars, bool mont,
                           thread_pool_t* da_pool = nullptr)
{
    // 点倍乘函数的封装，输入为点和标量的 slice，即指针+长度
    // 该函数使用了 slice_t 数据结构，可支持更加灵活的输入方式

    // 调用底层的点倍乘函数，将 slice 转换为指针并传入函数中
    mult_pippenger<bucket_t>(ret, points.data(),
                                  std::min(points.size(), scalars.size()),
                                  scalars.data(), mont, da_pool);
}
#endif
