// Copyright (C) 2019-2023 Aleo Systems Inc.
// This file is part of the snarkVM library.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[allow(unused_imports)]
use blst::*;

use core::ffi::c_void;
sppark::cuda_error!();

#[repr(C)]
pub enum NTTInputOutputOrder {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

#[repr(C)]
pub enum NTTDirection {
    Forward = 0,
    Inverse = 1,
}

#[repr(C)]
pub enum NTTType {
    Standard = 0,
    Coset = 1,
}

extern "C" {
    fn snarkvm_ntt(
        inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> cuda::Error;

    fn snarkvm_polymul(
        out: *mut core::ffi::c_void,
        pcount: usize,
        polynomials: *const core::ffi::c_void,
        plens: *const core::ffi::c_void,
        ecount: usize,
        evaluations: *const core::ffi::c_void,
        elens: *const core::ffi::c_void,
        lg_domain_size: u32,
    ) -> cuda::Error;

    fn snarkvm_msm(
        out: *mut c_void,
        points_with_infinity: *const c_void,
        npoints: usize,
        scalars: *const c_void,
        ffi_affine_sz: usize,
    ) -> cuda::Error;
}

///////////////////////////////////////////////////////////////////////////////
// Rust functions
///////////////////////////////////////////////////////////////////////////////

/// Compute an in-place NTT on the input data.
#[allow(non_snake_case)]
pub fn NTT<T>(
    domain_size: usize,
    inout: &mut [T],
    ntt_order: NTTInputOutputOrder,
    ntt_direction: NTTDirection,
    ntt_type: NTTType,
) -> Result<(), cuda::Error> {
    if (domain_size & (domain_size - 1)) != 0 {
        panic!("domain_size is not power of 2");
    }
    let lg_domain_size = domain_size.trailing_zeros();

    let err = unsafe {
        snarkvm_ntt(inout.as_mut_ptr() as *mut core::ffi::c_void, lg_domain_size, ntt_order, ntt_direction, ntt_type)
    };

    if err.code != 0 {
        return Err(err);
    }
    Ok(())
}

/// Compute a polynomial multiply
pub fn polymul<T: std::clone::Clone>(
    domain: usize,
    polynomials: &Vec<Vec<T>>,
    evaluations: &Vec<Vec<T>>,
    zero: &T,
) -> Result<Vec<T>, cuda::Error> {
    let initial_domain_size = domain;
    if (initial_domain_size & (initial_domain_size - 1)) != 0 {
        panic!("domain_size is not power of 2");
    }

    let lg_domain_size = initial_domain_size.trailing_zeros();

    let mut pptrs = Vec::new();
    let mut plens = Vec::new();
    for polynomial in polynomials {
        pptrs.push(polynomial.as_ptr() as *const core::ffi::c_void);
        plens.push(polynomial.len());
    }
    let mut eptrs = Vec::new();
    let mut elens = Vec::new();
    for evaluation in evaluations {
        eptrs.push(evaluation.as_ptr() as *const core::ffi::c_void);
        elens.push(evaluation.len());
    }

    let mut out = Vec::new();
    out.resize(initial_domain_size, zero.clone());
    let err = unsafe {
        snarkvm_polymul(
            out.as_mut_ptr() as *mut core::ffi::c_void,
            pptrs.len(),
            pptrs.as_ptr() as *const core::ffi::c_void,
            plens.as_ptr() as *const core::ffi::c_void,
            eptrs.len(),
            eptrs.as_ptr() as *const core::ffi::c_void,
            elens.as_ptr() as *const core::ffi::c_void,
            lg_domain_size,
        )
    };

    if err.code != 0 {
        return Err(err);
    }
    Ok(out)
}

/// Compute a multi-scalar multiplication
pub fn msm<Affine, Projective, Scalar>(points: &[Affine], scalars: &[Scalar]) -> Result<Projective, cuda::Error> {
    let npoints = scalars.len();  // 获取标量集合的长度，即点集合的数量
    if npoints > points.len() {
        panic!("length mismatch {} points < {} scalars", npoints, scalars.len())  // 如果点集合的数量小于标量集合的数量，则抛出错误
    }
    //println!("rust msm points {}, scalars {}", points.len(),scalars.len());

    // 创建一个未初始化的 Projective 结构体实例，用于存储计算结果
    #[allow(clippy::uninit_assumed_init)]
    let mut ret: Projective = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

    // 调用 FFI 函数 snarkvm_msm 进行变基 MSM 计算
    let err = unsafe {
        snarkvm_msm(
            &mut ret as *mut _ as *mut c_void,    // 存储计算结果的指针
            points as *const _ as *const c_void,   // 点集合的指针
            npoints,                               // 点集合的数量
            scalars as *const _ as *const c_void,  // 标量集合的指针
            std::mem::size_of::<Affine>(),         // 每个点的大小（以字节为单位）
        )
    };
    // 如果调用返回的错误码不为0，则返回错误信息
    if err.code != 0 {
        return Err(err);
    }

    // 返回计算结果
    Ok(ret)
}
