//! SynthSeg label tables.
//!
//! One table per model generation. The entries are ordered by ascending FreeSurfer label id,
//! which is the channel order of the network's output, so index `c` of a posterior map is
//! label [`SynthSegLabels::ids`]`[c]`.
//!
//! Transcribed from `data/labels_classes_priors/` in the SynthSeg distribution (Apache-2.0),
//! with `flip` and `topology` pre-resolved from `SynthSeg/predict.py:get_flip_indices` and the
//! topological-class file so the runtime does no table building.

/// Which SynthSeg generation a set of weights belongs to.
///
/// The generations differ in their label set — 2.0 adds a general `csf` class (FreeSurfer id
/// 24) — so the weights and the table have to agree.
#[cfg_attr(feature = "introspection", derive(serde::Serialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum SynthSegVersion {
    /// SynthSeg 1.0 (`synthseg_1.0.h5`), 32 labels. The default because its weights ship in
    /// the SynthSeg repository itself, so the export is reproducible from an Apache-2.0 source.
    #[default]
    V1,
    /// SynthSeg 2.0 (`synthseg_2.0.h5`), 33 labels — adds `csf`. The better model, but its
    /// weights are a separate download (FreeSurfer, or the link in the SynthSeg README).
    V2,
}

/// The label table for one SynthSeg generation.
pub struct SynthSegLabels {
    /// FreeSurfer label ids, ascending; index = network output channel.
    pub ids: &'static [i32],
    /// Human-readable structure names, parallel to [`ids`](Self::ids).
    pub names: &'static [&'static str],
    /// Channel permutation applied to the left-right flipped pass before averaging: channel `c`
    /// of the flipped posteriors corresponds to channel `flip[c]` of the unflipped ones.
    pub flip: &'static [usize],
    /// Topological class per channel. Channels sharing a non-zero class are cleaned up together
    /// by keeping the largest connected component of their union; class 0 is exempt.
    pub topology: &'static [u32],
}

impl SynthSegVersion {
    /// The label table for this generation.
    pub fn labels(self) -> &'static SynthSegLabels {
        match self {
            SynthSegVersion::V1 => &V1,
            SynthSegVersion::V2 => &V2,
        }
    }

    /// Number of network output channels (= number of labels, background included).
    pub fn n_labels(self) -> usize {
        self.labels().ids.len()
    }
}

static V1: SynthSegLabels = SynthSegLabels {
    ids: &[
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43, 44, 46, 47,
        49, 50, 51, 52, 53, 54, 58, 60,
    ],
    names: &[
        "background",
        "left cerebral white matter",
        "left cerebral cortex",
        "left lateral ventricle",
        "left inferior lateral ventricle",
        "left cerebellum white matter",
        "left cerebellum cortex",
        "left thalamus",
        "left caudate",
        "left putamen",
        "left pallidum",
        "3rd ventricle",
        "4th ventricle",
        "brain-stem",
        "left hippocampus",
        "left amygdala",
        "left accumbens area",
        "left ventral DC",
        "right cerebral white matter",
        "right cerebral cortex",
        "right lateral ventricle",
        "right inferior lateral ventricle",
        "right cerebellum white matter",
        "right cerebellum cortex",
        "right thalamus",
        "right caudate",
        "right putamen",
        "right pallidum",
        "right hippocampus",
        "right amygdala",
        "right accumbens area",
        "right ventral DC",
    ],
    flip: &[
        0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 11, 12, 13, 28, 29, 30, 31, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 14, 15, 16, 17,
    ],
    topology: &[
        0, 4, 4, 4, 4, 5, 5, 6, 7, 8, 9, 1, 2, 3, 10, 11, 12, 13, 14, 14, 14, 14, 15, 15, 16, 17,
        18, 19, 20, 21, 22, 23,
    ],
};

static V2: SynthSegLabels = SynthSegLabels {
    ids: &[
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46,
        47, 49, 50, 51, 52, 53, 54, 58, 60,
    ],
    names: &[
        "background",
        "left cerebral white matter",
        "left cerebral cortex",
        "left lateral ventricle",
        "left inferior lateral ventricle",
        "left cerebellum white matter",
        "left cerebellum cortex",
        "left thalamus",
        "left caudate",
        "left putamen",
        "left pallidum",
        "3rd ventricle",
        "4th ventricle",
        "brain-stem",
        "left hippocampus",
        "left amygdala",
        "csf",
        "left accumbens area",
        "left ventral DC",
        "right cerebral white matter",
        "right cerebral cortex",
        "right lateral ventricle",
        "right inferior lateral ventricle",
        "right cerebellum white matter",
        "right cerebellum cortex",
        "right thalamus",
        "right caudate",
        "right putamen",
        "right pallidum",
        "right hippocampus",
        "right amygdala",
        "right accumbens area",
        "right ventral DC",
    ],
    flip: &[
        0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 11, 12, 13, 29, 30, 16, 31, 32, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 14, 15, 17, 18,
    ],
    topology: &[
        0, 4, 4, 4, 4, 5, 5, 6, 7, 8, 9, 1, 2, 3, 10, 11, 0, 12, 13, 14, 14, 14, 14, 15, 15, 16,
        17, 18, 19, 20, 21, 22, 23,
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    fn check(v: SynthSegVersion) {
        let l = v.labels();
        let n = l.ids.len();
        assert_eq!(l.names.len(), n, "names length");
        assert_eq!(l.flip.len(), n, "flip length");
        assert_eq!(l.topology.len(), n, "topology length");
        assert!(l.ids.windows(2).all(|w| w[0] < w[1]), "ids must be strictly ascending");
        // Flipping twice is the identity, and background never moves.
        assert_eq!(l.flip[0], 0);
        for c in 0..n {
            assert_eq!(l.flip[l.flip[c]], c, "flip is not an involution at {c}");
        }
        // Every left structure is paired with the matching right one.
        for (c, name) in l.names.iter().enumerate() {
            if let Some(rest) = name.strip_prefix("left ") {
                assert_eq!(l.names[l.flip[c]], format!("right {rest}"));
            }
        }
    }

    #[test]
    fn v1_table_is_self_consistent() {
        check(SynthSegVersion::V1);
        assert_eq!(SynthSegVersion::V1.n_labels(), 32);
    }

    #[test]
    fn v2_table_is_self_consistent() {
        check(SynthSegVersion::V2);
        assert_eq!(SynthSegVersion::V2.n_labels(), 33);
        // 2.0's addition over 1.0.
        assert!(SynthSegVersion::V2.labels().ids.contains(&24));
        assert!(!SynthSegVersion::V1.labels().ids.contains(&24));
    }
}
