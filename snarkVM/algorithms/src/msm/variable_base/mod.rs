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

pub mod batched;
pub mod standard;

#[cfg(target_arch = "x86_64")]
pub mod prefetch;

use snarkvm_curves::{bls12_377::G1Affine, traits::AffineCurve};
use snarkvm_fields::PrimeField;

use core::any::TypeId;

pub struct VariableBase;

impl VariableBase {
    pub fn msm<G: AffineCurve>(
        bases: &[G],                                            // 输入参数：点集合，作为批处理加法的基点
        scalars: &[<G::ScalarField as PrimeField>::BigInteger], // 输入参数：标量集合，与基点对应的标量值
    ) -> G::Projective {
        // 对于 BLS12-377 曲线，我们使用批处理加法技术进行变基 MSM 计算。
        if TypeId::of::<G>() == TypeId::of::<G1Affine>() {
            #[cfg(all(feature = "cuda", target_arch = "x86_64"))]
            // TODO SNP: 设置阈值
            if scalars.len() > 1024 {
                let result = snarkvm_algorithms_cuda::msm::<G, G::Projective, <G::ScalarField as PrimeField>::BigInteger>(
                    bases, scalars,
                );  // 使用 CUDA 实现的 MSM 算法进行计算
                if let Ok(result) = result {
                    return result;
                }
            }
            batched::msm(bases, scalars)  // 使用批处理技术进行变基 MSM 计算
        }
        // 对于其他曲线，我们使用 Pippenger's 算法进行变基 MSM 计算。
        else {
            standard::msm(bases, scalars)  // 使用 Pippenger's 算法进行变基 MSM 计算
        }
    }

    #[cfg(test)]
    fn msm_naive<G: AffineCurve>(
        bases: &[G],                                            // 输入参数：点集合，作为批处理加法的基点
        scalars: &[<G::ScalarField as PrimeField>::BigInteger], // 输入参数：标量集合，与基点对应的标量值
    ) -> G::Projective {
        use itertools::Itertools;
        use snarkvm_utilities::BitIteratorBE;

        bases.iter().zip_eq(scalars).map(|(base, scalar)| base.mul_bits(BitIteratorBE::new(*scalar))).sum()
    }

    #[cfg(test)]
    fn msm_naive_parallel<G: AffineCurve>(
        bases: &[G],                                            // 输入参数：点集合，作为批处理加法的基点
        scalars: &[<G::ScalarField as PrimeField>::BigInteger], // 输入参数：标量集合，与基点对应的标量值
    ) -> G::Projective {
        use rayon::prelude::*;
        use snarkvm_utilities::BitIteratorBE;

        bases.par_iter().zip_eq(scalars).map(|(base, scalar)| base.mul_bits(BitIteratorBE::new(*scalar))).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snarkvm_curves::bls12_377::{Fr, G1Affine};
    use snarkvm_fields::PrimeField;
    use snarkvm_utilities::rand::TestRng;

    #[cfg(all(feature = "cuda", target_arch = "x86_64"))]
    use snarkvm_curves::ProjectiveCurve;

    fn create_scalar_bases<G: AffineCurve<ScalarField = F>, F: PrimeField>(
        rng: &mut TestRng,
        size: usize,
    ) -> (Vec<G>, Vec<F::BigInteger>) {
        let bases = (0..size).map(|_| G::rand(rng)).collect::<Vec<_>>();
        let scalars = (0..size).map(|_| F::rand(rng).to_bigint()).collect::<Vec<_>>();
        (bases, scalars)
    }

    #[test]
    fn test_msm() {
        use snarkvm_curves::ProjectiveCurve;
        for msm_size in [1, 5, 10, 50, 100, 500, 1000] {
            let mut rng = TestRng::default();
            let (bases, scalars) = create_scalar_bases::<G1Affine, Fr>(&mut rng, msm_size);

            let naive_a = VariableBase::msm_naive(bases.as_slice(), scalars.as_slice()).to_affine();
            let naive_b = VariableBase::msm_naive_parallel(bases.as_slice(), scalars.as_slice()).to_affine();
            assert_eq!(naive_a, naive_b, "MSM size: {msm_size}");

            let candidate = standard::msm(bases.as_slice(), scalars.as_slice()).to_affine();
            assert_eq!(naive_a, candidate, "MSM size: {msm_size}");

            let candidate = batched::msm(bases.as_slice(), scalars.as_slice()).to_affine();
            assert_eq!(naive_a, candidate, "MSM size: {msm_size}");
        }
    }

    #[cfg(all(feature = "cuda", target_arch = "x86_64"))]
    #[test]
    fn test_msm_cuda() {
        let mut rng = TestRng::default();
        for i in 2..17 {
            let (bases, scalars) = create_scalar_bases::<G1Affine, Fr>(&mut rng, 1 << i);
            let rust = standard::msm(bases.as_slice(), scalars.as_slice());
            let cuda = VariableBase::msm::<G1Affine>(bases.as_slice(), scalars.as_slice());
            assert_eq!(rust.to_affine(), cuda.to_affine());
        }
    }
}
