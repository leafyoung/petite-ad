//! Batch data types for evaluating compiled graphs on multiple inputs.

use crate::{AutodiffError, Result};

/// Flat row-major batch input view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchInputs<'a> {
    /// Flat row-major input data.
    pub data: &'a [f64],
    /// Number of rows in the batch.
    pub batch_size: usize,
    /// Number of inputs per row.
    pub input_dim: usize,
}

impl<'a> BatchInputs<'a> {
    /// Create a validated row-major input view.
    pub fn new(data: &'a [f64], batch_size: usize, input_dim: usize) -> Result<Self> {
        if data.len() != batch_size.saturating_mul(input_dim) {
            return Err(AutodiffError::InvalidGraph {
                reason: "batch data length must equal batch_size * input_dim",
            });
        }
        Ok(Self {
            data,
            batch_size,
            input_dim,
        })
    }

    /// Return one input row.
    ///
    /// Prefer [`BatchInputs::try_row`] when invalid row indices should return an error.
    #[must_use]
    pub fn row(&self, index: usize) -> &[f64] {
        self.try_row(index).expect("batch row index out of bounds")
    }

    /// Return one input row, or an error when `index` is out of range.
    pub fn try_row(&self, index: usize) -> Result<&[f64]> {
        if index >= self.batch_size {
            return Err(AutodiffError::IndexOutOfBounds {
                index,
                max_index: self.batch_size.saturating_sub(1),
            });
        }
        let start = index * self.input_dim;
        Ok(&self.data[start..start + self.input_dim])
    }
}

/// Flat row-major batch values.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchValues {
    pub data: Vec<f64>,
    pub batch_size: usize,
    pub output_dim: usize,
}

/// Reusable flat row-major batch value buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchValuesBuffer {
    pub data: Vec<f64>,
    pub batch_size: usize,
    pub output_dim: usize,
}

impl BatchValuesBuffer {
    /// Create an empty reusable output buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reset(&mut self, batch_size: usize, output_dim: usize) {
        self.data.clear();
        self.data.reserve(batch_size.saturating_mul(output_dim));
        self.batch_size = batch_size;
        self.output_dim = output_dim;
    }

    /// Clone the current buffer contents into an owned result value.
    #[must_use]
    pub fn to_values(&self) -> BatchValues {
        BatchValues {
            data: self.data.clone(),
            batch_size: self.batch_size,
            output_dim: self.output_dim,
        }
    }
}

/// Flat row-major batch scalar-output gradients.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchGradients {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
    pub batch_size: usize,
    pub input_dim: usize,
}

/// Reusable flat row-major batch scalar-output gradient buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchGradientsBuffer {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
    pub batch_size: usize,
    pub input_dim: usize,
}

impl BatchGradientsBuffer {
    /// Create an empty reusable gradient buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reset(&mut self, batch_size: usize, input_dim: usize) {
        self.values.clear();
        self.gradients.clear();
        self.values.reserve(batch_size);
        self.gradients.reserve(batch_size.saturating_mul(input_dim));
        self.batch_size = batch_size;
        self.input_dim = input_dim;
    }

    /// Clone the current buffer contents into an owned result value.
    #[must_use]
    pub fn to_gradients(&self) -> BatchGradients {
        BatchGradients {
            values: self.values.clone(),
            gradients: self.gradients.clone(),
            batch_size: self.batch_size,
            input_dim: self.input_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_batch_inputs_try_row_valid() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let batch = BatchInputs::new(&data, 2, 2).unwrap();
        let row0 = batch.try_row(0).unwrap();
        assert_eq!(row0, &[1.0, 2.0]);
        let row1 = batch.try_row(1).unwrap();
        assert_eq!(row1, &[3.0, 4.0]);
    }

    #[test]
    fn test_batch_inputs_try_row_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let batch = BatchInputs::new(&data, 2, 2).unwrap();
        assert!(batch.try_row(2).is_err());
        assert!(batch.try_row(100).is_err());
    }

    #[test]
    fn test_batch_inputs_invalid_shape() {
        assert!(BatchInputs::new(&[1.0, 2.0], 2, 2).is_err());
        assert!(BatchInputs::new(&[1.0, 2.0, 3.0], 1, 2).is_err());
    }

    #[test]
    fn test_batch_values_buffer_reuse() {
        // We can't test with CompiledGraph here — just test buffer lifecycle.
        let mut buffer = BatchValuesBuffer::new();

        buffer.reset(1, 1);
        buffer.data.push(3.0);
        let values = buffer.to_values();
        assert_eq!(values.batch_size, 1);
        assert_eq!(values.data, &[3.0]);

        buffer.reset(2, 2);
        buffer.data.extend_from_slice(&[9.0, 13.0]);
        let values2 = buffer.to_values();
        assert_eq!(values2.batch_size, 2);
        assert_eq!(values2.data, &[9.0, 13.0]);
    }

    #[test]
    fn test_batch_gradients_buffer_reuse() {
        let mut buffer = BatchGradientsBuffer::new();

        buffer.reset(1, 2);
        buffer.values.push(3.0);
        buffer.gradients.extend_from_slice(&[1.0, 1.0]);
        let grad = buffer.to_gradients();
        assert_eq!(grad.batch_size, 1);
        assert_eq!(grad.values, &[3.0]);
        assert_eq!(grad.gradients, &[1.0, 1.0]);

        buffer.reset(2, 2);
        buffer.values.push(9.0);
        buffer.values.push(13.0);
        buffer.gradients.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let grad2 = buffer.to_gradients();
        assert_eq!(grad2.batch_size, 2);
        assert_eq!(grad2.values, &[9.0, 13.0]);
        assert_eq!(grad2.gradients, &[1.0, 1.0, 1.0, 1.0]);
    }
}
