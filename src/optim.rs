//! Small optimizer utilities independent from graph construction.

use crate::{AutodiffError, Result};

fn check_lengths(params: &[f64], grads: &[f64]) -> Result<()> {
    if params.len() == grads.len() {
        Ok(())
    } else {
        Err(AutodiffError::InvalidArguments {
            reason: "parameter and gradient lengths must match",
        })
    }
}

/// Plain gradient descent optimizer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientDescent {
    /// Learning rate multiplier.
    pub learning_rate: f64,
}

impl GradientDescent {
    /// Apply one in-place gradient descent step.
    pub fn step(&self, params: &mut [f64], grads: &[f64]) -> Result<()> {
        check_lengths(params, grads)?;
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= self.learning_rate * grad;
        }
        Ok(())
    }
}

/// Adam optimizer with explicit state.
#[derive(Debug, Clone, PartialEq)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    step: usize,
}

impl Adam {
    /// Create Adam state for `parameter_count` scalar parameters.
    ///
    /// # Panics
    ///
    /// Panics if beta1, beta2 are outside [0, 1) or epsilon <= 0.
    #[must_use]
    pub fn new(parameter_count: usize, learning_rate: f64) -> Self {
        Self::with_params(parameter_count, learning_rate, 0.9, 0.999, 1e-8)
    }

    /// Create Adam state with custom hyperparameters.
    ///
    /// # Panics
    ///
    /// Panics if beta1, beta2 are outside [0, 1) or epsilon <= 0.
    #[must_use]
    pub fn with_params(
        parameter_count: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        assert!(
            (0.0..1.0).contains(&beta1),
            "beta1 must be in [0, 1), got {}",
            beta1
        );
        assert!(
            (0.0..1.0).contains(&beta2),
            "beta2 must be in [0, 1), got {}",
            beta2
        );
        assert!(epsilon > 0.0, "epsilon must be > 0, got {}", epsilon);
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: vec![0.0; parameter_count],
            v: vec![0.0; parameter_count],
            step: 0,
        }
    }

    /// Apply one in-place Adam update.
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) -> Result<()> {
        check_lengths(params, grads)?;
        check_lengths(&self.m, grads)?;
        self.step += 1;
        let bias1 = 1.0 - self.beta1.powf(self.step as f64);
        let bias2 = 1.0 - self.beta2.powf(self.step as f64);

        for index in 0..params.len() {
            self.m[index] = self.beta1 * self.m[index] + (1.0 - self.beta1) * grads[index];
            self.v[index] =
                self.beta2 * self.v[index] + (1.0 - self.beta2) * grads[index] * grads[index];
            let m_hat = self.m[index] / bias1;
            let v_hat = self.v[index] / bias2;
            params[index] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_step() {
        let optimizer = GradientDescent { learning_rate: 0.1 };
        let mut params = [1.0, 2.0];
        optimizer.step(&mut params, &[0.5, -0.5]).unwrap();
        assert!((params[0] - 0.95).abs() < 1e-12);
        assert!((params[1] - 2.05).abs() < 1e-12);
    }

    #[test]
    fn test_adam_step_changes_params() {
        let mut optimizer = Adam::new(1, 0.1);
        let mut params = [0.0];
        optimizer.step(&mut params, &[-1.0]).unwrap();
        assert!(params[0] > 0.0);
    }

    // --- Additional tests for uncovered lines ---

    #[test]
    fn test_gradient_descent_length_mismatch() {
        let optimizer = GradientDescent { learning_rate: 0.1 };
        let mut params = [1.0];
        let result = optimizer.step(&mut params, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_adam_new_defaults() {
        let adam = Adam::new(3, 0.01);
        assert_eq!(adam.learning_rate, 0.01);
        assert!((adam.beta1 - 0.9).abs() < 1e-10);
        assert!((adam.beta2 - 0.999).abs() < 1e-10);
        assert!((adam.epsilon - 1e-8).abs() < 1e-20);
        assert_eq!(adam.m.len(), 3);
        assert_eq!(adam.v.len(), 3);
        assert_eq!(adam.step, 0);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut adam = Adam::new(1, 0.1);
        let mut params = [1.0];
        // Run several steps to exercise powf with increasing step count
        for _ in 0..5 {
            adam.step(&mut params, &[1.0]).unwrap();
        }
        // After 5 steps of gradient=1.0 starting from 1.0, params should decrease
        assert!(params[0] < 1.0);
    }

    #[test]
    fn test_adam_length_mismatch_params() {
        let mut adam = Adam::new(2, 0.1);
        let mut params = [1.0];
        assert!(adam.step(&mut params, &[1.0]).is_err());
    }

    #[test]
    fn test_adam_length_mismatch_grads() {
        let mut adam = Adam::new(1, 0.1);
        let mut params = [1.0];
        assert!(adam.step(&mut params, &[1.0, 2.0]).is_err());
    }
}
