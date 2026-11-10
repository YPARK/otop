//! A fixed-length history of samples, for the charts.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How many samples each history keeps.
pub const HISTORY: usize = 600;

/// Samples in percent, oldest first, each with when it was taken, since
/// the interval between them can change.
#[derive(Default, Clone)]
pub struct History(VecDeque<(Instant, f64)>);

impl History {
    pub fn push(&mut self, v: f64) {
        if self.0.len() == HISTORY {
            self.0.pop_front();
        }
        self.0.push_back((Instant::now(), v));
    }

    /// The last `n` samples, at most, oldest first.
    pub fn last(&self, n: usize) -> impl Iterator<Item = f64> + '_ {
        self.0
            .iter()
            .skip(self.0.len().saturating_sub(n))
            .map(|&(_, v)| v)
    }

    /// How long ago the oldest of the last `n` samples was taken.
    pub fn span(&self, n: usize) -> Duration {
        let skip = self.0.len().saturating_sub(n);
        self.0
            .get(skip)
            .map_or(Duration::ZERO, |&(t, _)| t.elapsed())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}
