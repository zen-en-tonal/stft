use crate::{variables::Q, windows};
use num_traits::{Float, FloatConst};

#[derive(Debug)]
pub struct Framing<T> {
    window: Vec<T>,
    q: Q,
    buffer: Vec<T>,
}

impl<T> Framing<T>
where
    T: Float + FloatConst,
{
    pub fn with_window(q: Q, window: impl Fn(T) -> T) -> Self {
        let window = windows::window(q, window);
        Self {
            window,
            buffer: vec![T::zero(); q.window_size()],
            q,
        }
    }

    pub fn process<F>(&mut self, signal: &mut [T], mut work_with_frame: F)
    where
        F: FnMut(&mut [T]),
    {
        let mut window_overlaps = vec![T::zero(); signal.len()];

        let num_frames: usize = signal.len() / self.q.hop_size() + 1;
        let mut m: usize = 0;

        while m < num_frames {
            let pos = m * self.q.hop_size();

            // TODO: self.buffer should be a ring buffer.
            self.buffer.rotate_left(self.q.hop_size());
            for n in 0..self.q.hop_size() {
                let pos_hop = self.q.window_size() - self.q.hop_size() + n;
                self.buffer[pos_hop] = *signal.get(pos + n).unwrap_or(&T::zero());
            }

            let mut windowed_frame: Vec<T> = self
                .buffer
                .iter()
                .zip(self.window.iter())
                .map(|(x, w)| *x * *w)
                .collect();

            work_with_frame(&mut windowed_frame);

            for iota in 0..self.q.window_size() {
                let sig_index = pos as i32 - self.q.overlap_size() as i32 + iota as i32;

                let has_signal = (0..signal.len() as i32).contains(&sig_index);
                let y = if has_signal {
                    // Sefety: The boundary is already checked.
                    unsafe { signal.get_unchecked_mut(sig_index as usize) }
                } else {
                    continue;
                };

                let w = self.window[iota];
                let x = windowed_frame[iota] * w;
                let is_on_overlap = iota < self.q.overlap_size();
                *y = if is_on_overlap { *y + x } else { x };

                window_overlaps[sig_index as usize] = window_overlaps[sig_index as usize] + w * w;
            }

            m += 1;
        }

        signal
            .iter_mut()
            .zip(&window_overlaps)
            .filter(|(_, scale)| **scale > T::zero())
            .for_each(|(y, scale)| *y = *y / *scale);
    }
}

#[cfg(test)]
mod tests {
    use super::Framing;
    use crate::{variables::Q, windows};

    #[test]
    fn framing() {
        let q = Q::new(2, 0.5);
        let mut framing = Framing::with_window(q, |_| 1.);
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
        let q = Q::new(2, 0.5);
        let mut framing = Framing::with_window(q, |_| 1.);
        let mut data: Vec<f32> = vec![1337.];
        let mut progress: Vec<Vec<f32>> = vec![];
        framing.process(&mut data, |v| progress.push(v.to_vec()));

        assert_eq!(vec![0., 0., 1337., 0.], progress[0]);
        assert_eq!(vec![1337.], data);
    }

    #[test]
    fn reconstruction() {
        let mut framing = Framing::with_window(Q::new(4, 0.75), windows::hann());
        let mut in_data: Vec<f32> = (1..64).map(|x| x as f32).collect();
        let out_data = in_data.clone();
        framing.process(&mut in_data, |_| {});
        assert_eq!(
            out_data,
            in_data.iter().map(|x| x.round()).collect::<Vec<_>>()
        );
    }
}
