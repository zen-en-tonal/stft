use crate::{frame::Framing, variables::Q, windows};
use num_complex::Complex;
use num_traits::{Float, FloatConst, Zero};
use rustfft::{Fft, FftNum, FftPlanner};
use std::sync::Arc;

pub struct Stft<T> {
    framing: Framing<T>,
    buffer: Vec<Complex<T>>,
    forward: Arc<dyn Fft<T>>,
    backward: Arc<dyn Fft<T>>,
    q: Q,
}

impl<T> Stft<T>
where
    T: Float + FloatConst + FftNum,
{
    pub fn new(window_level: usize, slide_size: usize, window: impl Fn(T) -> T) -> Stft<T> {
        let q = Q::new(window_level, slide_size);
        let mut planner = FftPlanner::new();
        Stft {
            q,
            framing: Framing::with_window(q, window),
            buffer: vec![Complex::zero(); q.window_size()],
            forward: planner.plan_fft_forward(q.window_size()),
            backward: planner.plan_fft_inverse(q.window_size()),
        }
    }

    pub fn with_hann(window_level: usize, slide_size: usize) -> Stft<T> {
        Stft::new(window_level, slide_size, windows::hann())
    }

    pub fn with_hamming(window_level: usize, slide_size: usize) -> Stft<T> {
        Stft::new(window_level, slide_size, windows::hamming())
    }

    pub fn window_size(&self) -> usize {
        self.q.window_size()
    }

    pub fn fft_size(&self) -> usize {
        self.window_size() / 2 + 1
    }

    pub fn shift_size(&self) -> usize {
        self.q.frame_shift()
    }

    pub fn process<F>(&mut self, real: &mut [T], mut closure: F)
    where
        F: FnMut(&mut [Complex<T>]),
    {
        let window_size = T::from(self.window_size()).unwrap();
        let fft_size = self.fft_size();

        self.framing.process(real, |buffer| {
            self.buffer
                .iter_mut()
                .zip(buffer.iter())
                .for_each(|(c, r)| {
                    c.re = *r;
                    c.im = T::zero();
                });

            self.forward.process(&mut self.buffer);

            closure(&mut self.buffer);

            for n in 1..(fft_size - 1) {
                self.buffer[fft_size - 1 + n] = self.buffer[fft_size - 1 - n].conj();
            }

            self.backward.process(&mut self.buffer);

            buffer
                .iter_mut()
                .zip(self.buffer.iter())
                .for_each(|(r, c)| *r = c.re / window_size);
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::Stft;

    #[test]
    fn basic() {
        let mut stft: Stft<f32> = Stft::with_hann(4, 2);
        let mut in_data = vec![10.; 128];
        let out_data = in_data.clone();
        stft.process(&mut in_data, |_| {});
        assert_eq!(
            in_data.iter().map(|x| x.round()).collect::<Vec<_>>(),
            out_data
        );
    }
}
