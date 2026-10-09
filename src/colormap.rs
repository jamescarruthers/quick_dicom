//! Pseudo-colour maps for greyscale images. Each has a twin in
//! `common.wgsl`; the two must agree, since exported videos are drawn here
//! and the screen is drawn there.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Colormap {
    #[default]
    Grey,
    /// Black, red, yellow, white. Common for nuclear medicine and PET.
    HotIron,
    /// Black through blue, green and yellow to red and white.
    Rainbow,
}

/// Evenly spaced stops of the rainbow map, from 0 to 1.
const RAINBOW: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
];

impl Colormap {
    pub const ALL: [Colormap; 3] = [Colormap::Grey, Colormap::HotIron, Colormap::Rainbow];

    pub fn name(self) -> &'static str {
        match self {
            Colormap::Grey => "Grey",
            Colormap::HotIron => "Hot iron",
            Colormap::Rainbow => "Rainbow",
        }
    }

    /// The number the shaders use for this map.
    pub fn shader_id(self) -> f32 {
        match self {
            Colormap::Grey => 0.0,
            Colormap::HotIron => 1.0,
            Colormap::Rainbow => 2.0,
        }
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&c| c == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Colour (each channel 0 to 1) for a displayed brightness `g` (0 to 1).
    pub fn apply(self, g: f32) -> [f32; 3] {
        let g = g.clamp(0.0, 1.0);
        match self {
            Colormap::Grey => [g; 3],
            Colormap::HotIron => [
                (g * 3.0).clamp(0.0, 1.0),
                (g * 3.0 - 1.0).clamp(0.0, 1.0),
                (g * 3.0 - 2.0).clamp(0.0, 1.0),
            ],
            Colormap::Rainbow => {
                let x = g * 7.0;
                let i = (x.floor() as usize).min(6);
                let f = x - i as f32;
                let (a, b) = (RAINBOW[i], RAINBOW[i + 1]);
                std::array::from_fn(|c| a[c] + (b[c] - a[c]) * f)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_run_from_black_to_white() {
        for map in Colormap::ALL {
            assert_eq!(map.apply(0.0), [0.0; 3], "{map:?}");
            assert_eq!(map.apply(1.0), [1.0; 3], "{map:?}");
        }
        assert_eq!(Colormap::HotIron.apply(0.5), [1.0, 0.5, 0.0]);
        assert_eq!(Colormap::Rainbow.apply(4.0 / 7.0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn next_cycles_through_all() {
        assert_eq!(Colormap::Rainbow.next(), Colormap::Grey);
        assert_eq!(Colormap::Grey.next(), Colormap::HotIron);
    }
}
