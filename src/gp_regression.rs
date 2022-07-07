use std::error;
use std::fmt;

use num_traits::{Float, FloatConst};

use rulinalg::{
    matrix::{BaseMatrix, Matrix},
    norm::*,
    vector::Vector,
};

//// KERNEL

pub trait CovarianceFunc<T: Clone> {
    fn apply(&self, x: &Vector<T>, y: &Vector<T>) -> T;
    fn apply_mm(&self, a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
        assert_eq!(a.cols(), b.rows());

        Matrix::from_fn(a.rows(), b.cols(), |row, col| {
            self.apply(&a.row(row).into(), &b.col(col).into())
        })
    }
    fn apply_mv(&self, m: &Matrix<T>, v: &Vector<T>) -> Vector<T> {
        assert_eq!(m.cols(), v.size());

        Vector::from_fn(m.rows(), |row| self.apply(&m.row(row).into(), v))
    }
}

pub struct RBF<T> {
    length_scale: T,
    variance: T,
}

impl<T> RBF<T> {
    pub fn new(length_scale: T, variance: T) -> Self {
        Self {
            length_scale,
            variance,
        }
    }
}

impl<T: Float> CovarianceFunc<T> for RBF<T> {
    fn apply(&self, x: &Vector<T>, y: &Vector<T>) -> T {
        self.variance
            * (-(x.metric(y, Euclidean) / self.length_scale).powi(2) / T::from(2).unwrap()).exp()
    }
}

//// MEAN FUNCTION

pub trait MeanFunc<T: Clone> {
    fn apply(&self, x: &Vector<T>) -> T;
    fn apply_m(&self, m: &Matrix<T>) -> Vector<T> {
        Vector::from_fn(m.rows(), |row| self.apply(&m.row(row).into()))
    }
}

pub struct ConstMean<T> {
    c: T,
}

impl<T> ConstMean<T> {
    pub fn new(c: T) -> Self {
        Self { c }
    }
}

impl<T: Float> MeanFunc<T> for ConstMean<T> {
    fn apply(&self, _x: &Vector<T>) -> T {
        self.c
    }
}

// GAUSSIAN PROCESS REGRESSOR

pub struct GaussianProcessRegressor<T, M, K> {
    mean: M,
    kernel: K,
    noise: T,
    params: Option<GPRegressionParams<T>>,
}

impl<T, M, K> GaussianProcessRegressor<T, M, K>
where
    T: 'static + Float + FloatConst + std::iter::Sum<T>,
    M: MeanFunc<T>,
    K: CovarianceFunc<T>,
{
    pub fn new(mean: M, cov: K, noise: T) -> Self {
        Self {
            mean,
            kernel: cov,
            noise,
            params: None,
        }
    }

    pub fn fit(
        &mut self,
        x_train: &Matrix<T>,
        y_train: &Vector<T>,
    ) -> Result<(), rulinalg::error::Error> {
        let k = self.kernel.apply_mm(&x_train, &x_train.transpose());

        let l = (&k + Matrix::identity(y_train.size()) * self.noise).cholesky()?;

        let alpha = l.transpose().solve_u_triangular(
            l.solve_l_triangular(y_train.clone() - self.mean.apply_m(x_train))?,
        )?;

        self.params = Some(GPRegressionParams {
            x_train: x_train.clone(),
            y_train: y_train.clone(),
            l,
            alpha,
        });

        Ok(())
    }

    pub fn predict(&self, x_test: &Vector<T>) -> Result<(T, T, T), ModelUntrainedError> {
        self.params
            .as_ref()
            .map(|params| {
                let k_star = self.kernel.apply_mv(&params.x_train, &x_test);

                let mean = self.mean.apply(&x_test) + k_star.dot(&params.alpha);

                let v = params.l.solve_l_triangular(k_star.clone()).unwrap();
                let variance = self.kernel.apply(&x_test, &x_test) - v.norm(Euclidean);

                // - 1 / 2 * (y^T \alpha + log |K| + n * log(2 * pi))
                let occam_factor = params.l.diag().map(|i| i.ln()).sum();
                let two = T::from(2f64).unwrap();
                let log_likelihood = -params.y_train.dot(&params.alpha) / two
                    - occam_factor
                    - (T::from(params.y_train.size()).unwrap()) * (two * T::PI()).ln() / two;

                (mean, variance, log_likelihood)
            })
            .ok_or(ModelUntrainedError)
    }
}

// Helper struct
struct GPRegressionParams<T> {
    x_train: Matrix<T>,
    y_train: Vector<T>,
    l: Matrix<T>,
    alpha: Vector<T>,
}

#[derive(Debug, Clone)]
pub struct ModelUntrainedError;

impl fmt::Display for ModelUntrainedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cannot use model when untrained")
    }
}

impl error::Error for ModelUntrainedError {}
