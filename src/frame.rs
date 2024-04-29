use crate::{variables::Q, windows};
use num_traits::{Float, FloatConst};

pub struct Framing<T> {
    analysis_window: Vec<T>,
    q: Q,
}

impl<T> Framing<T>
where
    T: Float + FloatConst,
{
    pub fn with_window(q: Q, window: impl Fn(T) -> T) -> Self {
        let window = windows::window(q, window);
        Self {
            analysis_window: window,
            q,
        }
    }

    pub fn process<F>(&self, signal: &mut [T], mut closure: F)
    where
        F: FnMut(&mut [T]),
    {
        let mut scratch = vec![T::zero(); signal.len()];
        let mut frame: Vec<T> = vec![T::zero(); self.q.window_size()];
        let half: i32 = (self.q.window_size() / 2) as i32;
        let mut center: i32 = 0;
        let mut scales: Vec<T> = vec![T::zero(); signal.len()];

        loop {
            if center > signal.len() as i32 {
                break;
            }

            let indice = ((center - half)..(center + half)).enumerate();

            indice.clone().for_each(|(frame_index, signal_index)| {
                let is_in_range = 0 <= signal_index && signal_index < signal.len() as i32;
                frame[frame_index] = if !is_in_range {
                    T::zero()
                } else {
                    let signal_index: usize = signal_index as usize;
                    signal[signal_index] * self.analysis_window[frame_index]
                };
            });

            closure(frame.as_mut_slice());

            indice.for_each(|(frame_index, signal_index)| {
                let is_in_range = 0 <= signal_index && signal_index < signal.len() as i32;
                if is_in_range {
                    let signal_index: usize = signal_index as usize;
                    let w = self.analysis_window[frame_index];
                    scratch[signal_index] = scratch[signal_index] + frame[frame_index] * w;
                    scales[signal_index] = scales[signal_index] + w * w;
                }
            });

            center += self.q.frame_shift() as i32;
        }

        signal.clone_from_slice(&scratch);

        for (x, s) in signal.iter_mut().zip(scales) {
            if s > T::zero() {
                *x = *x / s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Framing;
    use crate::{variables::Q, windows};

    #[test]
    fn framing() {
        let q = Q::new(2, 2);
        let framing = Framing {
            analysis_window: vec![1.; q.window_size()],
            q,
        };
        let mut data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let mut progress: Vec<Vec<f32>> = vec![];
        framing.process(&mut data, |v| progress.push(v.to_vec()));

        assert_eq!(vec![0., 0., 0., 1.], progress[0]);
        assert_eq!(vec![0., 1., 2., 3.], progress[1]);
        assert_eq!(vec![2., 3., 0., 0.], progress[2]);

        assert_eq!(vec![0., 1., 2., 3.], data);
    }

    #[test]
    fn works_with_signal_too_few() {
        let q = Q::new(2, 2);
        let framing = Framing {
            analysis_window: vec![1.; q.window_size()],
            q,
        };
        let mut data: Vec<f32> = vec![1337.];
        let mut progress: Vec<Vec<f32>> = vec![];
        framing.process(&mut data, |v| progress.push(v.to_vec()));

        assert_eq!(vec![0., 0., 1337., 0.], progress[0]);
        assert_eq!(vec![1337.], data);
    }

    #[test]
    fn reconstruction() {
        let framing = Framing::with_window(Q::new(5, 4), windows::hann());
        let mut in_data: Vec<f32> = (0..1024).map(|x| x as f32).collect();
        let out_data = in_data.clone();
        framing.process(&mut in_data, |_| {});
        assert_eq!(
            out_data,
            in_data.iter().map(|x| x.round()).collect::<Vec<_>>()
        );
    }
}
