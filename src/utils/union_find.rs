//! Disjoint-set union with integer potentials.
//!
//! A plain weighted union-find, plus a per-element integer offset relative to its
//! parent. That extra field is what makes it useful for phase unwrapping: the
//! elements are voxels (or regions), the potential is the number of 2π wraps that
//! separates an element from its set representative, and merging two sets under a
//! constraint of the form "b must sit `delta` wraps above a" is an O(α) operation
//! instead of a walk over every member of the smaller set.
//!
//! Callers that only need connectivity can pass `delta = 0` throughout and ignore
//! the potentials.

/// Union-find over `0..n` with an integer potential attached to each element.
///
/// Writing `v[i]` for the (never materialised) absolute value of element `i`, the
/// structure maintains `potential[i] == v[i] - v[parent[i]]`, so [`find`] can
/// report `v[i] - v[root]` while compressing the path it walks.
///
/// [`find`]: UnionFind::find
pub(crate) struct UnionFind {
    parent: Vec<u32>,
    /// `potential[i] = v[i] - v[parent[i]]`; for a root this is 0.
    potential: Vec<i32>,
    size: Vec<u32>,
    /// Scratch buffer for path compression, kept to avoid an allocation per `find`.
    path: Vec<u32>,
}

impl UnionFind {
    /// Create `n` singleton sets, each its own root with potential 0.
    pub(crate) fn new(n: usize) -> Self {
        debug_assert!(n <= u32::MAX as usize, "UnionFind is indexed by u32");
        Self {
            parent: (0..n as u32).collect(),
            potential: vec![0; n],
            size: vec![1; n],
            path: Vec::new(),
        }
    }

    /// Find the representative of `i`, returning `(root, v[i] - v[root])`.
    ///
    /// Compresses the path walked, so the potentials of every element on it are
    /// rewritten to be relative to the root.
    pub(crate) fn find(&mut self, i: usize) -> (usize, i32) {
        self.path.clear();
        let mut cur = i as u32;
        while self.parent[cur as usize] != cur {
            self.path.push(cur);
            cur = self.parent[cur as usize];
        }
        let root = cur;

        // Walk back down from the child of the root. `acc` accumulates the
        // potential of each visited element relative to the root.
        let mut acc = 0i32;
        for idx in (0..self.path.len()).rev() {
            let node = self.path[idx] as usize;
            acc += self.potential[node];
            self.potential[node] = acc;
            self.parent[node] = root;
        }

        (root as usize, acc)
    }

    /// Number of elements in the set containing `i`.
    // Part of the DSU surface rather than a current caller's need: best-path
    // unwrapping never asks for set sizes, PRELUDE's region merging will.
    #[allow(dead_code)]
    pub(crate) fn set_size(&mut self, i: usize) -> usize {
        let (root, _) = self.find(i);
        self.size[root] as usize
    }

    /// Join the sets of `a` and `b` under the constraint `v[b] - v[a] == delta`.
    ///
    /// Returns `false` (changing nothing) when the two are already joined — the
    /// existing potentials then already imply some difference, which may or may
    /// not agree with `delta`. Callers that care about inconsistent constraints
    /// should compare the potentials themselves before calling.
    ///
    /// The smaller set is attached under the larger; on a tie, `b`'s root is
    /// attached under `a`'s. Tree shape has no effect on the resulting potential
    /// *differences*, but it does decide which element ends up as the root, and
    /// hence the arbitrary constant that `find` reports potentials against.
    pub(crate) fn union_with_delta(&mut self, a: usize, b: usize, delta: i32) -> bool {
        let (ra, pa) = self.find(a);
        let (rb, pb) = self.find(b);
        if ra == rb {
            return false;
        }

        // v[b] - v[a] = (pb + v[rb]) - (pa + v[ra]) = delta
        //   =>  v[rb] - v[ra] = delta - pb + pa
        let d = delta - pb + pa;

        if self.size[ra] >= self.size[rb] {
            self.parent[rb] = ra as u32;
            self.potential[rb] = d;
            self.size[ra] += self.size[rb];
        } else {
            self.parent[ra] = rb as u32;
            self.potential[ra] = -d;
            self.size[rb] += self.size[ra];
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singletons_are_their_own_roots() {
        let mut uf = UnionFind::new(4);
        for i in 0..4 {
            assert_eq!(uf.find(i), (i, 0));
            assert_eq!(uf.set_size(i), 1);
        }
    }

    #[test]
    fn potentials_accumulate_along_a_chain() {
        let mut uf = UnionFind::new(5);
        // v[1]-v[0]=1, v[2]-v[1]=1, v[3]-v[2]=1, v[4]-v[3]=1
        for i in 0..4 {
            assert!(uf.union_with_delta(i, i + 1, 1));
        }
        let (root, p0) = uf.find(0);
        for i in 0..5 {
            let (r, p) = uf.find(i);
            assert_eq!(r, root);
            assert_eq!(p - p0, i as i32, "element {i} should sit {i} above element 0");
        }
        assert_eq!(uf.set_size(2), 5);
    }

    #[test]
    fn negative_and_mixed_deltas() {
        let mut uf = UnionFind::new(3);
        assert!(uf.union_with_delta(0, 1, -2));
        assert!(uf.union_with_delta(1, 2, 5));
        let (_, p0) = uf.find(0);
        let (_, p1) = uf.find(1);
        let (_, p2) = uf.find(2);
        assert_eq!(p1 - p0, -2);
        assert_eq!(p2 - p1, 5);
        assert_eq!(p2 - p0, 3);
    }

    #[test]
    fn joining_an_existing_set_is_a_no_op() {
        let mut uf = UnionFind::new(3);
        assert!(uf.union_with_delta(0, 1, 1));
        assert!(uf.union_with_delta(1, 2, 1));
        // 0 and 2 are already connected; the second constraint is ignored, even
        // though it contradicts the first.
        assert!(!uf.union_with_delta(0, 2, 99));
        let (_, p0) = uf.find(0);
        let (_, p2) = uf.find(2);
        assert_eq!(p2 - p0, 2);
    }

    #[test]
    fn merging_two_trees_preserves_internal_differences() {
        let mut uf = UnionFind::new(6);
        uf.union_with_delta(0, 1, 3);
        uf.union_with_delta(1, 2, 3);
        uf.union_with_delta(3, 4, -1);
        uf.union_with_delta(4, 5, -1);
        assert!(uf.union_with_delta(2, 3, 10));

        let p = |uf: &mut UnionFind, i: usize| uf.find(i).1;
        let p0 = p(&mut uf, 0);
        assert_eq!(p(&mut uf, 1) - p0, 3);
        assert_eq!(p(&mut uf, 2) - p0, 6);
        assert_eq!(p(&mut uf, 3) - p0, 16);
        assert_eq!(p(&mut uf, 4) - p0, 15);
        assert_eq!(p(&mut uf, 5) - p0, 14);
        assert_eq!(uf.set_size(0), 6);
    }

    #[test]
    fn deep_chain_does_not_overflow_the_stack() {
        // Path compression is iterative, so a degenerate chain is fine.
        let n = 200_000;
        let mut uf = UnionFind::new(n);
        for i in 0..n - 1 {
            uf.union_with_delta(i, i + 1, 0);
        }
        assert_eq!(uf.set_size(0), n);
        assert_eq!(uf.find(n - 1).1, 0);
    }
}
