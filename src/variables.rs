#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q {
    window_size: usize,
    hop_size: usize,
}

impl Q {
    pub fn new(window_order: usize, overlap_rate: f32) -> Self {
        assert!(window_order > 0);
        assert!(0. <= overlap_rate && overlap_rate < 1.);
        let window_size = 1 << window_order;
        let hop_size = window_size as f32 * (1. - overlap_rate);
        Self {
            window_size,
            hop_size: hop_size.floor() as usize,
        }
    }

    pub fn as_usize(&self) -> usize {
        self.window_size / self.hop_size
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    pub fn overlap_size(&self) -> usize {
        self.window_size() - self.hop_size()
    }
}

impl From<Q> for usize {
    fn from(value: Q) -> Self {
        value.as_usize()
    }
}
