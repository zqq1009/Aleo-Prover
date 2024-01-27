// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_GPU_T_CUH__
#define __SPPARK_UTIL_GPU_T_CUH__

#ifndef __CUDACC__
# include <cuda_runtime.h>
#endif

#include "thread_pool_t.hpp"
#include "exception.cuh"
#include "slice_t.hpp"

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

class gpu_t;
// GPU 类型（未提供定义）

size_t ngpus();
// 返回系统中GPU的数量

const gpu_t& select_gpu(int id = 0);
// 选择要使用的GPU，id 表示GPU的索引，默认为 0

const cudaDeviceProp& gpu_props(int id = 0);
// 返回指定 GPU 的设备属性，id 表示 GPU 的索引，默认为 0

const std::vector<const gpu_t*>& all_gpus();
// 返回所有 GPU 的列表

extern "C" bool cuda_available();
// 检查 CUDA 是否可用的外部 C 函数声明

class event_t {
    cudaEvent_t event; // CUDA 事件对象
public:
    event_t() : event(nullptr)
    {   CUDA_OK(cudaEventCreate(&event, cudaEventDisableTiming));   }
    // 构造函数，创建 CUDA 事件对象并禁用计时

    event_t(cudaStream_t stream) : event(nullptr)
    {
        CUDA_OK(cudaEventCreate(&event, cudaEventDisableTiming));
        CUDA_OK(cudaEventRecord(event, stream));
    }
    // 带流参数的构造函数，创建 CUDA 事件对象、记录事件并与流关联

    ~event_t()
    {   if (event) cudaEventDestroy(event);   }
    // 析构函数，销毁 CUDA 事件对象

    inline operator decltype(event)() const
    {   return event;   }
    // 类型转换运算符，返回事件对象

    inline void record(cudaStream_t stream)
    {   CUDA_OK(cudaEventRecord(event, stream));   }
    // 记录事件与指定流相关联

    inline void wait(cudaStream_t stream)
    {   CUDA_OK(cudaStreamWaitEvent(stream, event));   }
    // 等待指定流中与事件相关联的操作完成
};

struct launch_params_t {
    dim3 gridDim, blockDim; // 格子维度和线程块维度
    size_t shared; // 共享内存大小

    launch_params_t(dim3 g, dim3 b, size_t sz = 0) : gridDim(g), blockDim(b), shared(sz) {}
    // 构造函数，初始化格子维度、线程块维度和共享内存大小

    launch_params_t(int g, int b, size_t sz = 0) : gridDim(g), blockDim(b), shared(sz) {}
    // 构造函数，初始化格子维度、线程块维度和共享内存大小
};

class stream_t {
    cudaStream_t stream; // CUDA流对象
    const int gpu_id; // GPU设备ID
public:
    stream_t(int id) : gpu_id(id)
    {   cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);   } // 创建CUDA流对象，并设置为非阻塞流
    ~stream_t()
    {   cudaStreamDestroy(stream);   } // 销毁CUDA流对象
    inline operator decltype(stream)() const    { return stream; } // 将stream_t对象转换为cudaStream_t类型
    inline int id() const                       { return gpu_id; } // 获取GPU设备ID
    inline operator int() const                 { return gpu_id; } // 将stream_t对象转换为int类型

    inline void* Dmalloc(size_t sz) const
    {   void *d_ptr;
        CUDA_OK(cudaMallocAsync(&d_ptr, sz, stream)); // 在设备上异步分配内存
        return d_ptr;
    }
    inline void Dfree(void* d_ptr) const
    {   CUDA_OK(cudaFreeAsync(d_ptr, stream));   } // 在设备上异步释放内存

    template<typename T>
    inline void HtoD(T* dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   if (sz == sizeof(T))
            CUDA_OK(cudaMemcpyAsync(dst, src, nelems*sizeof(T),
                                    cudaMemcpyHostToDevice, stream)); // 在设备上异步将主机内存数据复制到设备内存
        else
            CUDA_OK(cudaMemcpy2DAsync(dst, sizeof(T), src, sz,
                                      std::min(sizeof(T), sz), nelems,
                                      cudaMemcpyHostToDevice, stream)); // 在设备上异步将主机内存数据复制到设备内存（二维内存复制）
    }
    template<typename T>
    inline void HtoD(T& dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   HtoD(&dst, src, nelems, sz);   } // 在设备上异步将主机内存数据复制到设备内存
    template<typename T, typename U>
    inline void HtoD(T& dst, const std::vector<U>& src,
                     size_t sz = sizeof(T)) const
    {   HtoD(&dst, &src[0], src.size(), sz);   } // 在设备上异步将主机内存数据复制到设备内存
    template<typename T, typename U>
    inline void HtoD(T* dst, const std::vector<U>& src,
                     size_t sz = sizeof(T)) const
    {   HtoD(dst, &src[0], src.size(), sz);   } // 在设备上异步将主机内存数据复制到设备内存
    template<typename T, typename U>
    inline void HtoD(T& dst, slice_t<U> src, size_t sz = sizeof(T)) const
    {   HtoD(&dst, &src[0], src.size(), sz);   } // 在设备上异步将主机内存数据复制到设备内存
    template<typename T, typename U>
    inline void HtoD(T* dst, slice_t<U> src, size_t sz = sizeof(T)) const
    {   HtoD(dst, &src[0], src.size(), sz);   } // 在设备上异步将主机内存数据复制到设备内存

    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim,
                                                size_t shared_sz,
                            Types... args) const
    {
        if (gpu_props(gpu_id).sharedMemPerBlock < shared_sz)
            CUDA_OK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_sz)); // 设置函数的最大动态共享内存大小
        if (gridDim.x == 0 || blockDim.x == 0) {
            int blockSize, minGridSize;

            CUDA_OK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, f)); // 获取函数的最大线程块大小
            if (blockDim.x == 0) blockDim.x = blockSize; // 如果未指定线程块大小，则使用计算得到的最大线程块大小
            if (gridDim.x == 0)  gridDim.x = minGridSize; // 如果未指定网格大小，则使用计算得到的最小网格大小
        }
        void* va_args[sizeof...(args)] = { &args... };
        CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim,
                                            va_args, shared_sz, stream)); // 启动协作式GPU核函数
    }
    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), const launch_params_t& lps,
                            Types... args) const
    {   launch_coop(f, lps.gridDim, lps.blockDim, lps.shared, args...);   } // 启动协作式GPU核函数

    template<typename T>
    inline void DtoH(T* dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   if (sz == sizeof(T))
            CUDA_OK(cudaMemcpyAsync(dst, src, nelems*sizeof(T),
                                    cudaMemcpyDeviceToHost, stream)); // 在设备上异步将设备内存数据复制到主机内存
        else
            CUDA_OK(cudaMemcpy2DAsync(dst, sizeof(T), src, sz,
                                      std::min(sizeof(T), sz), nelems,
                                      cudaMemcpyDeviceToHost, stream)); // 在设备上异步将设备内存数据复制到主机内存（二维内存复制）
    }
    template<typename T>
    inline void DtoH(T& dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   DtoH(&dst, src, nelems, sz);   } // 在设备上异步将设备内存数据复制到主机内存
    template<typename T>
    inline void DtoH(std::vector<T>& dst, const void* src,
                     size_t sz = sizeof(T)) const
    {   DtoH(&dst[0], src, dst.size(), sz);   } // 在设备上异步将设备内存数据复制到主机内存

    inline void sync() const
    {   CUDA_OK(cudaStreamSynchronize(stream));   } // 同步等待流中的所有操作完成

    inline void notify(cudaHostFn_t cb, void* data)
    {   CUDA_OK(cudaLaunchHostFunc(stream, cb, data));   } // 在流中异步启动主机函数

    template<class T>
    inline void notify(T& sema)
    {   notify([](void* s) { reinterpret_cast<T*>(s)->notify(); }, &sema);   } // 在流中异步通知信号量

    inline void record(cudaEvent_t event)
    {   CUDA_OK(cudaEventRecord(event, stream));   } // 在流中记录事件
    inline void wait(cudaEvent_t event)
    {   CUDA_OK(cudaStreamWaitEvent(stream, event));   } // 在流中等待事件
};

class gpu_t {
public:
    static const size_t FLIP_FLOP = 3; // GPU 可以使用的翻转流的数量

private:
    int gpu_id, cuda_id; // GPU 的 id 和 CUDA 设备 id
    cudaDeviceProp prop; // 存储 GPU 设备属性的结构体
    size_t total_mem; // GPU 的总内存
    mutable stream_t zero = {gpu_id}; // 默认的流对象，用于同步操作
    mutable stream_t flipflop[FLIP_FLOP] = {gpu_id, gpu_id, gpu_id}; // 翻转流对象数组，用于并行操作
    mutable thread_pool_t pool{"SPPARK_GPU_T_AFFINITY"}; // 线程池对象

public:
    gpu_t(int id, int real_id, const cudaDeviceProp& p)
            : gpu_id(id), cuda_id(real_id), prop(p)
    {
        size_t freeMem;
        CUDA_OK(cudaMemGetInfo(&freeMem, &total_mem)); // 获取 GPU 的可用内存信息
    }

    inline int cid() const                  { return cuda_id; } // 返回 CUDA 设备 id
    inline int id() const                   { return gpu_id; } // 返回 GPU id
    inline operator int() const             { return gpu_id; } // 类型转换为 int
    inline const auto& props() const        { return prop; } // 返回 GPU 设备属性
    inline int sm_count() const             { return prop.multiProcessorCount; } // 返回 GPU 的 SM（Streaming Multiprocessor）数量
    inline void select() const              { cudaSetDevice(cuda_id); } // 选择当前 GPU
    stream_t& operator[](size_t i) const    { return flipflop[i%FLIP_FLOP]; } // 返回指定位置的翻转流对象
    inline operator stream_t&() const       { return zero; } // 类型转换为流对象
    inline operator cudaStream_t() const    { return zero; } // 类型转换为 CUDA 流对象

    inline size_t ncpus() const             { return pool.size(); } // 返回线程池的线程数量
    template<class Workable>
    inline void spawn(Workable work) const  { pool.spawn(work); } // 在线程池中执行任务
    template<class Workable>
    inline void par_map(size_t num_items, size_t stride, Workable work,
                        size_t max_workers = 0) const
    {   pool.par_map(num_items, stride, work, max_workers); } // 并行映射操作，将任务拆分给多个线程池中的线程进行处理

    inline void* Dmalloc(size_t sz) const // 在 GPU 上分配内存
    {
        void *d_ptr = zero.Dmalloc(sz); // 调用流对象的 Dmalloc 函数分配内存
        zero.sync(); // 同步流对象
        return d_ptr;
    }
    inline void Dfree(void* d_ptr) const // 在 GPU 上释放内存
    {
        zero.Dfree(d_ptr); // 调用流对象的 Dfree 函数释放内存
        zero.sync(); // 同步流对象
    }

    // 将主机内存数据复制到 GPU 内存
    template<typename T>
    inline void HtoD(T* dst, const void* src, size_t nelems, size_t sz = sizeof(T)) const
    {
        zero.HtoD(dst, src, nelems, sz); // 调用流对象的 HtoD 函数将数据从主机内存复制到 GPU 内存
    }

    // 将主机内存数据复制到 GPU 内存
    template<typename T>
    inline void HtoD(T& dst, const void* src, size_t nelems, size_t sz = sizeof(T)) const
    {
        HtoD(&dst, src, nelems, sz); // 调用上面定义的第一个函数，将地址和参数传递给第一个函数进行处理
    }

    // 将主机内存中的向量数据复制到 GPU 内存
    template<typename T, typename U>
    inline void HtoD(T& dst, const std::vector<U>& src, size_t sz = sizeof(T)) const
    {
        HtoD(&dst, &src[0], src.size(), sz); // 调用上面定义的第一个函数，将向量数据的地址和参数传递给第一个函数进行处理
    }


    // 启动协作的 GPU 核函数
    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim,
                            size_t shared_sz, Types... args) const
    {
        zero.launch_coop(f, gridDim, blockDim, shared_sz, args...); // 调用流对象的 launch_coop 函数启动协作的 GPU 核函数
    }

    // 启动协作的 GPU 核函数
    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), const launch_params_t& lps,
                            Types... args) const
    {
        zero.launch_coop(f, lps, args...); // 调用流对象的 launch_coop 函数启动协作的 GPU 核函数
    }

    // 将 GPU 内存数据复制到主机内存
    template<typename T>
    inline void DtoH(T* dst, const void* src, size_t nelems, size_t sz = sizeof(T)) const
    {
        zero.DtoH(dst, src, nelems, sz); // 调用流对象的 DtoH 函数将数据从 GPU 内存复制到主机内存
    }

    // 将 GPU 内存数据复制到主机内存
    template<typename T>
    inline void DtoH(T& dst, const void* src, size_t nelems, size_t sz = sizeof(T)) const
    {
        DtoH(&dst, src, nelems, sz); // 调用上面定义的第一个函数，将地址和参数传递给第一个函数进行处理
    }

    // 将 GPU 内存中的向量数据复制到主机内存
    template<typename T>
    inline void DtoH(std::vector<T>& dst, const void* src, size_t sz = sizeof(T)) const
    {
        DtoH(&dst[0], src, dst.size(), sz); // 调用上面定义的第一个函数，将向量数据的地址和参数传递给第一个函数进行处理
    }


    inline void sync() const
    {
        zero.sync(); // 同步默认的流对象
        for (auto& f : flipflop)
            f.sync(); // 同步翻转流对象
    }
};

template<typename T> class gpu_ptr_t {
    // 内部结构体，用于管理 GPU 指针内存
    struct inner {
        T* ptr; // GPU 指针
        std::atomic<size_t> ref_cnt; // 引用计数器，记录指向此 GPU 指针的对象数量
        int real_id; // 记录该指针所在的设备 ID
        inline inner(T* p) : ptr(p), ref_cnt(1)
        {   cudaGetDevice(&real_id);   } // 构造函数，初始化 GPU 指针和引用计数器，记录所在设备 ID
    };
    inner *ptr; // 内部指针，指向 inner 类型的对象

public:
    // 默认构造函数，初始化内部指针为空
    gpu_ptr_t() : ptr(nullptr)    {}
    // 构造函数，传入 GPU 指针，创建一个新的 inner 对象
    gpu_ptr_t(T* p)               { ptr = new inner(p); }
    // 复制构造函数，将 r 的内部指针赋值给当前对象的内部指针，并将引用计数加 1
    gpu_ptr_t(const gpu_ptr_t& r) { *this = r; }
    // 析构函数，释放内存并减少引用计数
    ~gpu_ptr_t()
    {
        if (ptr && ptr->ref_cnt.fetch_sub(1, std::memory_order_seq_cst) == 1) {
            int current_id;
            cudaGetDevice(&current_id);
            if (current_id != ptr->real_id)
                cudaSetDevice(ptr->real_id); // 如果当前设备 ID 不是指针所在设备 ID，切换到指针所在设备
            cudaFree(ptr->ptr); // 释放 GPU 指针内存
            if (current_id != ptr->real_id)
                cudaSetDevice(current_id); // 切换回当前设备
            delete ptr; // 释放 inner 对象
        }
    }

    // 复制赋值运算符，将 r 的内部指针赋值给当前对象的内部指针，并将引用计数加 1
    gpu_ptr_t& operator=(const gpu_ptr_t& r)
    {
        if (this != &r)
            (ptr = r.ptr)->ref_cnt.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }
    // 移动赋值运算符，将 r 的内部指针赋值给当前对象的内部指针，并将 r 的内部指针置为空
    gpu_ptr_t& operator=(gpu_ptr_t&& r) noexcept
    {
        if (this != &r) {
            ptr = r.ptr;
            r.ptr = nullptr;
        }
        return *this;
    }

    // 类型转换运算符，将当前对象转换为 T 类型的指针
    inline operator T*() const                  { return ptr->ptr; }

    // 简化 FFI 返回值过程，通过 by_value 结构体包装 inner 指针，便于返回值拷贝
    using by_value = struct { inner *ptr; };
    // 类型转换运算符，将当前对象转换为 by_value 结构体类型
    operator by_value() const
    {   ptr->ref_cnt.fetch_add(1, std::memory_order_relaxed); return {ptr};   }
    // 构造函数，通过 by_value 结构体类型初始化内部指针
    gpu_ptr_t(by_value v)   { ptr = v.ptr; }
};


// A simple way to allocate a temporary device pointer without having to
// care about freeing it.
// 设备内存指针类模板
template<typename T> class dev_ptr_t {
    T* d_ptr; // 内部指针，指向设备上分配的内存

public:
    // 构造函数，分配 nelems 个 T 类型对象的内存，并将内部指针指向该内存
    dev_ptr_t(size_t nelems) : d_ptr(nullptr)
    {
        if (nelems) {
            // 将 nelems 向上对齐至 WARP_SZ 的倍数，WARP_SZ 定义为 32，加速访问
            size_t n = (nelems+WARP_SZ-1) & ((size_t)0-WARP_SZ);
            CUDA_OK(cudaMalloc(&d_ptr, n * sizeof(T))); // 在设备上分配内存
        }
    }

    // 带流参数的构造函数，与上述构造函数类似，但在异步流上分配内存
    dev_ptr_t(size_t nelems, stream_t& s) : d_ptr(nullptr)
    {
        if (nelems) {
            size_t n = (nelems+WARP_SZ-1) & ((size_t)0-WARP_SZ);
            CUDA_OK(cudaMallocAsync(&d_ptr, n * sizeof(T), s)); // 在异步流上分配内存
        }
    }

    // 禁止拷贝构造函数和拷贝赋值运算符
    dev_ptr_t(const dev_ptr_t& r) = delete;
    dev_ptr_t& operator=(const dev_ptr_t& r) = delete;

    // 析构函数，释放设备上分配的内存
    ~dev_ptr_t() { if (d_ptr) cudaFree((void*)d_ptr); }

    // 类型转换运算符，将当前对象转换为 const T* 类型的指针
    inline operator const T*() const            { return d_ptr; }
    // 类型转换运算符，将当前对象转换为 T* 类型的指针
    inline operator T*() const                  { return d_ptr; }
    // 类型转换运算符，将当前对象转换为 void* 类型的指针
    inline operator void*() const               { return (void*)d_ptr; }
    // 重载数组下标运算符，返回设备上内存指定位置的 T 类型对象引用
    inline const T& operator[](size_t i) const  { return d_ptr[i]; }
    // 重载数组下标运算符，返回设备上内存指定位置的 T 类型对象引用
    inline T& operator[](size_t i)              { return d_ptr[i]; }
};


#endif
