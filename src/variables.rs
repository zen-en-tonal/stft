#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q {
    window_size: usize,
    frame_shift: usize,
}

impl Q {
    pub fn new(window_level: usize, frame_shift: usize) -> Self {
        assert!(window_level > 0);
        let window_size = 1 << window_level;
        assert!(frame_shift < window_size);
        Self {
            window_size,
            frame_shift,
        }
    }

    pub fn as_usize(&self) -> usize {
        self.window_size / self.frame_shift
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn frame_shift(&self) -> usize {
        self.frame_shift
    }
}

impl From<Q> for usize {
    fn from(value: Q) -> Self {
        value.as_usize()
    }
}
