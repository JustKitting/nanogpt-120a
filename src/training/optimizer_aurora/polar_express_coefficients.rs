#[derive(Clone, Copy)]
pub(super) struct PolarExpressCoefficients {
    pub(super) a: f32,
    pub(super) b: f32,
    pub(super) c: f32,
}

pub(super) fn polar_express_coefficients(iter: usize) -> PolarExpressCoefficients {
    let index = if iter < RAW_COEFFICIENTS.len() {
        iter
    } else {
        RAW_COEFFICIENTS.len() - 1
    };
    let coefficients = RAW_COEFFICIENTS[index];
    if index + 1 == RAW_COEFFICIENTS.len() {
        return coefficients;
    }

    PolarExpressCoefficients {
        a: coefficients.a / SAFETY,
        b: coefficients.b / SAFETY_CUBED,
        c: coefficients.c / SAFETY_QUINTIC,
    }
}

const SAFETY: f32 = 1.01;
const SAFETY_CUBED: f32 = 1.030301;
const SAFETY_QUINTIC: f32 = 1.0510101;

// Algorithm 1's degree-5 schedule. The safety scaling is applied to every
// listed tuple except the final Pade fallback, which is repeated as needed.
const RAW_COEFFICIENTS: [PolarExpressCoefficients; 8] = [
    PolarExpressCoefficients {
        a: 8.28721201814563,
        b: -23.595886519098837,
        c: 17.300387312530933,
    },
    PolarExpressCoefficients {
        a: 4.107059111542203,
        b: -2.9478499167379106,
        c: 0.5448431082926601,
    },
    PolarExpressCoefficients {
        a: 3.9486908534822946,
        b: -2.908902115962949,
        c: 0.5518191394370137,
    },
    PolarExpressCoefficients {
        a: 3.3184196573706015,
        b: -2.488488024314874,
        c: 0.51004894012372,
    },
    PolarExpressCoefficients {
        a: 2.300652019954817,
        b: -1.6689039845747493,
        c: 0.4188073119525673,
    },
    PolarExpressCoefficients {
        a: 1.891301407787398,
        b: -1.2679958271945868,
        c: 0.37680408948524835,
    },
    PolarExpressCoefficients {
        a: 1.8750014808534479,
        b: -1.2500016453999487,
        c: 0.3750001645474248,
    },
    PolarExpressCoefficients {
        a: 1.875,
        b: -1.25,
        c: 0.375,
    },
];
