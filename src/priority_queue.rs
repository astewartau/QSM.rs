/// Bucket-based priority queue for O(1) push/pop operations
/// Used for region growing phase unwrapping where priorities are discrete (0-255)
///
/// Higher priority = better quality = processed first (starts from highest bin)

pub struct BucketQueue<T> {
    bins: Vec<Vec<T>>,
    current_priority: isize,  // Can go negative when exhausted
    count: usize,
    n_bins: usize,
}

impl<T> BucketQueue<T> {
    pub fn new(n_bins: usize) -> Self {
        BucketQueue {
            bins: (0..n_bins).map(|_| Vec::new()).collect(),
            current_priority: (n_bins - 1) as isize,  // Start at highest priority
            count: 0,
            n_bins,
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn push(&mut self, priority: usize, item: T) {
        let priority = priority.min(self.n_bins - 1);
        self.bins[priority].push(item);
        self.count += 1;
        // Update current_priority if this is higher (better quality)
        if (priority as isize) > self.current_priority {
            self.current_priority = priority as isize;
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }

        // Find next non-empty bin starting from current priority (going DOWN)
        while self.current_priority >= 0 && self.bins[self.current_priority as usize].is_empty() {
            self.current_priority -= 1;
        }

        if self.current_priority < 0 {
            return None;
        }

        self.count -= 1;
        self.bins[self.current_priority as usize].pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_queue() {
        let mut queue: BucketQueue<(usize, usize, usize)> = BucketQueue::new(256);
        assert!(queue.is_empty());

        queue.push(100, (1, 2, 3));
        queue.push(50, (4, 5, 6));
        queue.push(200, (7, 8, 9));

        assert!(!queue.is_empty());

        // Should pop HIGHEST priority first (better quality edges first)
        assert_eq!(queue.pop(), Some((7, 8, 9)));  // priority 200
        assert_eq!(queue.pop(), Some((1, 2, 3)));  // priority 100
        assert_eq!(queue.pop(), Some((4, 5, 6)));  // priority 50
        assert!(queue.is_empty());
        assert_eq!(queue.pop(), None);
    }
}
