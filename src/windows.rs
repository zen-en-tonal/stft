use crate::variables::Q;
use num_traits::{Float, FloatConst};

pub fn window<T: Float>(q: Q, f: impl Fn(T) -> T) -> Vec<T> {
    (0..q.window_size())
        .map(|t| T::from(t).unwrap() / T::from(q.window_size()).unwrap())
        .map(f)
        .collect()
}

pub fn hamming<T: Float + FloatConst>() -> impl Fn(T) -> T {
    let a = T::from(0.54).unwrap();
    let b = T::from(0.46).unwrap();
    let tau = T::from(T::TAU()).unwrap();
    move |t| a - b * (tau * t).cos()
}

pub fn hann<T: Float + FloatConst>() -> impl Fn(T) -> T {
    let a = T::from(0.50).unwrap();
    let tau = T::from(T::TAU()).unwrap();
    move |t| a - a * (tau * t).cos()
}
