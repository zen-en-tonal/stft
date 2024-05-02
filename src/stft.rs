use crate::{frame::Framing, variables::Q, windows};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, FloatConst};
use rustfft::{Fft, FftNum, FftPlanner};
use std::sync::Arc;

pub struct Stft<T> {
    framing: Framing<T>,
    forward: Arc<dyn Fft<T>>,
    backward: Arc<dyn Fft<T>>,
    q: Q,
}

impl<T> Stft<T>
where
    T: Float + FloatConst + FftNum,
{
    pub fn new(q: Q, window: impl Fn(T) -> T) -> Stft<T> {
        let mut planner = FftPlanner::new();
        Stft {
            q,
            framing: Framing::with_window(q, window),
            forward: planner.plan_fft_forward(q.window_size()),
            backward: planner.plan_fft_inverse(q.window_size()),
        }
    }

    pub fn with_hann(q: Q) -> Stft<T> {
        Stft::new(q, windows::hann())
    }

    pub fn with_hamming(q: Q) -> Stft<T> {
        Stft::new(q, windows::hamming())
    }

    pub fn window_size(&self) -> usize {
        self.q.window_size()
    }

    pub fn fft_size(&self) -> usize {
        self.window_size() / 2 + 1
    }

    pub fn hop_size(&self) -> usize {
        self.q.hop_size()
    }

    pub fn process<F>(&mut self, real: &mut [T], works_with_spec: F)
    where
        F: Fn(&[Complex<T>]) -> Vec<Complex<T>> + Sync + Send,
    {
        let window_size = T::from(self.window_size()).unwrap();
        let fft_size = self.fft_size();

        self.framing.process(real, |frame| {
            // fill self.buffer with real.
            let mut buffer: Vec<Complex<T>> =
                frame.iter().map(|r| Complex::new(*r, T::zero())).collect();

            self.forward.process(&mut buffer);

            works_with_spec(&mut buffer[0..fft_size]);

            // restore mirror image.
            for n in 1..(fft_size - 1) {
                buffer[fft_size - 1 + n] = buffer[fft_size - 1 - n].conj();
            }

            self.backward.process(&mut buffer);

            buffer.into_iter().map(|c| c.re / window_size).collect()
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{Stft, Q};

    #[test]
    fn basic() {
        let mut stft: Stft<f32> = Stft::with_hamming(Q::new(10, 0.75));
        for _ in 0..10 {
            let mut in_data = vec![10.; 4098];
            let out_data = in_data.clone();
            stft.process(&mut in_data, |c| c.to_vec());
            assert_eq!(
                in_data.iter().map(|x| x.round()).collect::<Vec<_>>(),
                out_data
            );
        }
    }
}
