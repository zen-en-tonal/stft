use crate::variables::Q;
use num_traits::{Float, FloatConst};

pub fn window<T: Float>(q: Q, f: impl Fn(T) -> T) -> Vec<T> {
    (0..q.window_size())
        .map(|t| T::from(t).unwrap() / T::from(q.window_size()).unwrap())
        .map(f)
        .collect()
}

pub fn hamming<T: Float + FloatConst>(q: Q) -> Vec<T> {
    let a = T::from(0.54).unwrap();
    let b = T::from(0.46).unwrap();
    let tau = T::from(T::TAU()).unwrap();
    window(q, |t| a - b * (tau * t).cos())
}

pub fn hann<T: Float + FloatConst>(q: Q) -> Vec<T> {
    let a = T::from(0.50).unwrap();
    let tau = T::from(T::TAU()).unwrap();
    window(q, |t| a - a * (tau * t).cos())
}

/// creates a optimized synth window from an analysis window.
/// the algorithm is taken from [Eq. 22](https://www.jstage.jst.go.jp/article/jasj/72/12/72_764/_pdf).
pub fn synth_window<T: Float>(analysis_window: &[T], q: Q) -> Vec<T> {
    let mut synth_window = Vec::from(analysis_window);
    for t in 0..synth_window.len() {
        let range = ((1 - q.as_usize() as i32)..(q.as_usize() as i32 - 1))
            .map(|m| t as i32 + m * q.frame_shift() as i32);
        let mut den = T::zero();
        for i in range {
            let is_in_range = 0 <= i && i < analysis_window.len() as i32;
            if is_in_range {
                den = den + analysis_window[i as usize].powi(2);
            }
        }
        synth_window[t] = analysis_window[t] / den;
    }
    synth_window
}

#[cfg(test)]
mod tests {
    use crate::variables::Q;

    use super::{hamming, synth_window};

    #[test]
    fn hamming_synth() {
        let q = Q::new(2, 1);
        let ham = hamming::<f32>(q);
        synth_window(&ham, q);
    }
}
