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

use snarkvm_algorithms::msm::*;
use snarkvm_curves::AffineCurve;
use snarkvm_fields::PrimeField;
use snarkvm_utilities::TestRng;

use cactus_timer::{start_timer, end_timer};

fn create_scalar_bases<G: AffineCurve<ScalarField = F>, F: PrimeField>(size: usize) -> (Vec<G>, Vec<F::BigInteger>) {
    let mut rng = TestRng::default();

    let bases = std::iter::repeat((0..(size / 1000)).map(|_| G::rand(&mut rng)).collect::<Vec<_>>())
        .take(1000)
        .flatten()
        .collect::<Vec<_>>();
    let scalars = (0..size).map(|_| F::rand(&mut rng).to_bigint()).collect::<Vec<_>>();
    (bases, scalars)
}


fn main() {
    use snarkvm_curves::bls12_377::{Fr, G1Affine};
    let (bases, scalars) = create_scalar_bases::<G1Affine, Fr>(2000000);
    let size = 16384;
    //let size = 1025;

    let msm_timer = start_timer("");
    VariableBase::msm(&bases[..size], &scalars[..size]);
    end_timer(&msm_timer,"VariableBase::msm");
}
