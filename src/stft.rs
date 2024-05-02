use crate::{frame::Framing, variables::Q, windows};
use num_complex::Complex;
use num_traits::{Float, FloatConst, Zero};
use rustfft::{Fft, FftNum, FftPlanner};
use std::sync::Arc;

pub struct Stft<T> {
    framing: Framing<T>,
    buffer: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
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
            buffer: vec![Complex::zero(); q.window_size()],
            scratch: vec![Complex::zero(); q.window_size()],
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

    pub fn process<F>(&mut self, real: &mut [T], mut works_with_spec: F)
    where
        F: FnMut(&mut [Complex<T>]),
    {
        let window_size = T::from(self.window_size()).unwrap();
        let fft_size = self.fft_size();

        self.framing.process(real, |frame| {
            // fill self.buffer with real.
            self.buffer.iter_mut().zip(frame.iter()).for_each(|(c, r)| {
                c.re = *r;
                c.im = T::zero();
            });

            self.forward
                .process_with_scratch(&mut self.buffer, &mut self.scratch);

            works_with_spec(&mut self.buffer[0..fft_size]);

            // restore mirror image.
            for n in 1..(fft_size - 1) {
                self.buffer[fft_size - 1 + n] = self.buffer[fft_size - 1 - n].conj();
            }

            self.backward
                .process_with_scratch(&mut self.buffer, &mut self.scratch);

            frame
                .iter_mut()
                .zip(self.buffer.iter())
                .for_each(|(r, c)| *r = c.re / window_size);
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{Stft, Q};

    #[test]
    fn basic() {
        let mut stft: Stft<f32> = Stft::with_hann(Q::new(10, 0.75));
        for _ in 0..10 {
            let mut in_data = vec![10.; 4098];
            let out_data = in_data.clone();
            stft.process(&mut in_data, |_| {});
            assert_eq!(
                in_data.iter().map(|x| x.round()).collect::<Vec<_>>(),
                out_data
            );
        }
    }
}
