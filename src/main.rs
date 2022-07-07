#[macro_use]
extern crate rulinalg;

use rand::distributions::Uniform;
use rand::prelude::*;

mod gp_regression;
use gp_regression::*;

use rulinalg::{matrix::Matrix, norm::*, vector::Vector};

// TODO: Find better test data set
// TODO: Implement hyperparameter optimization
// TODO: Plot predictions from regression

fn func(x: f64) -> f64 {
    x * x.sin()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data

    let training_samples: usize = 50;
    let test_samples: usize = 1000;

    let x_lower_bound: f64 = -500.;
    let x_upper_bound: f64 = 500.;

    let mut rng = thread_rng();
    let dist = Uniform::new_inclusive(x_lower_bound, x_upper_bound);

    let x_train_vals: Vec<f64> = dist.sample_iter(&mut rng).take(training_samples).collect();
    let x_train = Matrix::new(training_samples, 1, &*x_train_vals);
    let y_train: Vector<f64> = (&x_train_vals).into_iter().map(|&x| func(x)).collect();

    // Train model

    let mut regressor = GaussianProcessRegressor::new(ConstMean::new(0.), RBF::new(1., 1.), 1e-10);
    regressor.fit(&x_train, &y_train)?;

    // Evaluate model

    let x_test_vals: Vec<f64> = dist.sample_iter(&mut rng).take(training_samples).collect();
    let y_test: Vector<f64> = (&x_test_vals).into_iter().map(|&x| func(x)).collect();
    let y_prediction: Vector<f64> = (&x_train_vals)
        .into_iter()
        .map(|&x| regressor.predict(&vector![x]).unwrap().0)
        .collect();

    let y_sqr_residuals = y_test.metric(&y_prediction, Euclidean);
    let y_abs_residuals = y_test.metric(&y_prediction, Lp::Integer(1));
    let y_sqr_mean_residual = (&y_test - &y_test.mean()).norm(Euclidean);

    let rs = 1. - y_sqr_residuals / y_sqr_mean_residual;
    let mse = y_sqr_residuals / test_samples as f64;
    let mae = y_abs_residuals / test_samples as f64;

    println!("R squared: {}", rs);
    println!("Mean squared error: {}", mse);
    println!("Mean absolute error: {}", mae);

    Ok(())
}
