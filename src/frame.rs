use crate::{variables::Q, windows};
use num_traits::{Float, FloatConst};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Framing<T> {
    window: Vec<T>,
    q: Q,
}

impl<T> Framing<T>
where
    T: Float + FloatConst + Sync + Send,
{
    pub fn with_window(q: Q, window: impl Fn(T) -> T) -> Self {
        let window = windows::window(q, window);
        Self { window, q }
    }

    pub fn process<F>(&self, signal: &mut [T], work_with_frame: F)
    where
        F: Fn(&[T]) -> Vec<T> + Sync + Send,
    {
        let mut window_overlaps = vec![T::zero(); signal.len()];

        let frames: Vec<Frame<T>> = self
            .split_into_frames(signal)
            .into_par_iter()
            .map(|frame| Frame {
                slice: work_with_frame(&frame.slice),
                ..frame
            })
            .collect();

        signal.fill(T::zero());

        for frame in frames {
            for (n, sample) in frame.slice.iter().enumerate() {
                let i = frame.from + n;
                let Some(y) = signal.get_mut(i) else {
                    continue;
                };
                let w = self.window[n];
                *y = *y + *sample * w;

                window_overlaps[i] = window_overlaps[i] + w * w;
            }
        }

        signal
            .iter_mut()
            .zip(&window_overlaps)
            .filter(|(_, scale)| **scale > T::zero())
            .for_each(|(y, scale)| *y = *y / *scale);
    }

    fn split_into_frames(&self, signal: &[T]) -> Vec<Frame<T>> {
        let num_frames: usize = signal.len() / self.q.hop_size() + 1;
        let mut frames: Vec<Frame<T>> = Vec::with_capacity(num_frames);

        for m in 0..num_frames {
            let pos = m * self.q.hop_size();

            let mut scratch = Vec::with_capacity(self.q.window_size());

            for (n, w) in self.window.iter().enumerate() {
                scratch.push(*signal.get(pos + n).unwrap_or(&T::zero()) * *w);
            }

            frames.push(Frame {
                slice: scratch,
                from: pos,
            })
        }

        frames
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Frame<T> {
    slice: Vec<T>,
    from: usize,
}

#[cfg(test)]
mod tests {
    use super::Framing;
    use crate::{variables::Q, windows};

    #[test]
    fn framing() {
        let q = Q::new(2, 0.5);
        let framing = Framing::with_window(q, |_| 1.);
        let frames = framing.split_into_frames(&vec![1., 2., 3., 4.]);

        assert_eq!(frames[0].slice, vec![1., 2., 3., 4.]);
        assert_eq!(frames[1].slice, vec![3., 4., 0., 0.]);
    }

    #[test]
    fn works_with_signal_too_few() {
        let q = Q::new(2, 0.5);
        let framing = Framing::with_window(q, |_| 1.);
        let mut data: Vec<f32> = vec![1337.];
        framing.process(&mut data, |v| v.to_vec());

        assert_eq!(vec![1337.], data);
    }

    #[test]
    fn reconstruction() {
        let q = Q::new(4, 0.75);
        let framing = Framing::with_window(q, windows::hamming());
        let mut in_data: Vec<f32> = (1..64).map(|x| x as f32).collect();
        let out_data = in_data.clone();
        framing.process(&mut in_data, |v| v.to_vec());
        assert_eq!(
            out_data,
            in_data.iter().map(|x| x.round()).collect::<Vec<_>>()
        );
    }
}
