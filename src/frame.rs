use crate::{variables::Q, windows};
use num_traits::{Float, FloatConst};

pub struct Framing<T> {
    analysis_window: Vec<T>,
    synth_window: Vec<T>,
    q: Q,
}

impl<T> Framing<T>
where
    T: Float + FloatConst,
{
    pub fn with_window(q: Q, window: impl Fn(T) -> T) -> Self {
        let window = windows::window(q, window);
        Self {
            synth_window: windows::synth_window(&window, q),
            analysis_window: window,
            q,
        }
    }

    pub fn hamming(q: Q) -> Self {
        let hamming = windows::hamming(q);
        Self {
            synth_window: windows::synth_window(&hamming, q),
            analysis_window: hamming,
            q,
        }
    }

    pub fn hann(q: Q) -> Self {
        let window = windows::hann(q);
        Self {
            synth_window: windows::synth_window(&window, q),
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
                    scratch[signal_index] =
                        scratch[signal_index] + frame[frame_index] * self.synth_window[frame_index];
                }
            });

            center += self.q.frame_shift() as i32;
        }

        signal.clone_from_slice(&scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::Framing;
    use crate::variables::Q;

    #[test]
    fn framing() {
        let q = Q::new(2, 2);
        let framing = Framing {
            analysis_window: vec![1.; q.window_size()],
            synth_window: vec![1.; q.window_size()],
            q,
        };
        let mut data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let mut progress: Vec<Vec<f32>> = vec![];
        framing.process(&mut data, |v| progress.push(v.to_vec()));

        assert_eq!(vec![0., 0., 0., 1.], progress[0]);
        assert_eq!(vec![0., 1., 2., 3.], progress[1]);
        assert_eq!(vec![2., 3., 0., 0.], progress[2]);

        assert_eq!(vec![0., 2., 4., 6.], data);
    }

    #[test]
    fn works_with_signal_too_few() {
        let q = Q::new(2, 2);
        let framing = Framing {
            analysis_window: vec![1.; q.window_size()],
            synth_window: vec![1.; q.window_size()],
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
        let framing = Framing::hann(Q::new(2, 1));
        let mut in_data: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let out_data = in_data.clone();
        framing.process(&mut in_data, |_| {});
        assert_eq!(
            out_data,
            in_data.iter().map(|x| x.floor()).collect::<Vec<_>>()
        );
    }
}
