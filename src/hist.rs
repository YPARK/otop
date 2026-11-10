//! A fixed-length history of samples, for the charts.

use std::collections::VecDeque;

/// How many samples each history keeps.
pub const HISTORY: usize = 600;

/// Samples in percent, oldest first.
#[derive(Default, Clone)]
pub struct History(VecDeque<f64>);

impl History {
    pub fn push(&mut self, v: f64) {
        if self.0.len() == HISTORY {
            self.0.pop_front();
        }
        self.0.push_back(v);
    }

    /// The last `n` samples, at most, oldest first.
    pub fn last(&self, n: usize) -> impl Iterator<Item = f64> + '_ {
        self.0.iter().skip(self.0.len().saturating_sub(n)).copied()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}
