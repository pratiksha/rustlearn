//! A machine learning crate for Rust.
//!
//!
//! # Introduction
//!
//! This crate contains reasonably effective implementations
//! of a number of common machine learing algorithms.
//!
//! At the moment, `rustlearn` uses its own basic dense and sparse array types, but I will be happy
//! to use something more robust once a clear winner in that space emerges.
//!
//! # Features
//!
//! ## Matrix primitives
//!
//! - [dense matrices](array/dense/index.html)
//! - [sparse matrices](array/sparse/index.html)
//!
//! ## Models
//!
//! - [logistic regression](linear_models/sgdclassifier/index.html) using stochastic gradient descent,
//! - [support vector machines](svm/libsvm/svc/index.html) using the `libsvm` library,
//! - [decision trees](trees/decision_tree/index.html) using the CART algorithm,
//! - [random forests](ensemble/random_forest/index.html) using CART decision trees, and
//! - [factorization machines](factorization/factorization_machines/index.html).
//!
//! All the models support fitting and prediction on both dense and sparse data, and the implementations
//! should be roughly competitive with Python `sklearn` implementations, both in accuracy and performance.
//!
//! ## Cross-validation
//!
//! - [k-fold cross-validation](cross_validation/cross_validation/index.html)
//! - [shuffle split](cross_validation/shuffle_split/index.html)
//!
//! ## Metrics
//!
//! - [accuracy](metrics/fn.accuracy_score.html)
//! - [mean_absolute_error](metrics/fn.mean_absolute_error.html)
//! - [mean_squared_error](metrics/fn.mean_squared_error.html)
//! - [ROC AUC score](metrics/ranking/fn.roc_auc_score.html)
//! - [dcg_score](metrics/ranking/fn.dcg_score.html)
//! - [ndcg_score](metrics/ranking/fn.ndcg_score.html)
//!
//! ## Parallelization
//!
//! A number of models support both parallel model fitting and prediction.
//!
//! ## Model serialization
//!
//! Model serialization is supported via `serde`.
//!
//! # Using `rustlearn`
//! Usage should be straightforward.
//!
//! - import the prelude for alll the linear algebra primitives and common traits:
//!
//! ```
//! use rustlearn::prelude::*;
//! ```
//!
//! - import individual models and utilities from submodules:
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! use rustlearn::linear_models::sgdclassifier::Hyperparameters;
//! // more imports
//! ```
//!
//! # Examples
//!
//! ## Logistic regression
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::datasets::iris;
//! use rustlearn::cross_validation::CrossValidation;
//! use rustlearn::linear_models::sgdclassifier::Hyperparameters;
//! use rustlearn::metrics::accuracy_score;
//!
//!
//! let (X, y) = iris::load_data();
//!
//! let num_splits = 10;
//! let num_epochs = 5;
//!
//! let mut accuracy = 0.0;
//!
//! for (train_idx, test_idx) in CrossValidation::new(X.rows(), num_splits) {
//!
//!     let X_train = X.get_rows(&train_idx);
//!     let y_train = y.get_rows(&train_idx);
//!     let X_test = X.get_rows(&test_idx);
//!     let y_test = y.get_rows(&test_idx);
//!
//!     let mut model = Hyperparameters::new(X.cols())
//!                                     .learning_rate(0.5)
//!                                     .l2_penalty(0.0)
//!                                     .l1_penalty(0.0)
//!                                     .one_vs_rest();
//!
//!     for _ in 0..num_epochs {
//!         model.fit(&X_train, &y_train).unwrap();
//!     }
//!
//!     let prediction = model.predict(&X_test).unwrap();
//!     accuracy += accuracy_score(&y_test, &prediction);
//! }
//!
//! accuracy /= num_splits as f32;
//!
//! ```
//!
//! ## Random forest
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! use rustlearn::ensemble::random_forest::Hyperparameters;
//! use rustlearn::datasets::iris;
//! use rustlearn::trees::decision_tree;
//!
//! let (data, target) = iris::load_data();
//!
//! let mut tree_params = decision_tree::Hyperparameters::new(data.cols());
//! tree_params.min_samples_split(10)
//!     .max_features(4);
//!
//! let mut model = Hyperparameters::new(tree_params, 10)
//!     .one_vs_rest();
//!
//! model.fit(&data, &target).unwrap();
//!
//! // Optionally serialize and deserialize the model
//!
//! // let encoded = bincode::serialize(&model).unwrap();
//! // let decoded: OneVsRestWrapper<RandomForest> = bincode::deserialize(&encoded).unwrap();
//!
//! let prediction = model.predict(&data).unwrap();
//! ```

// Only use unstable features when we are benchmarking
#![cfg_attr(feature = "bench", feature(test))]
// Allow conventional capital X for feature arrays.
#![allow(non_snake_case)]

#[cfg(feature = "bench")]
extern crate test;

#[cfg(test)]
extern crate bincode;

#[cfg(test)]
extern crate csv;

#[cfg(test)]
extern crate serde_json;

extern crate crossbeam;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate rsmalloc;

#[global_allocator]
static GLOBAL: rsmalloc::Allocator = rsmalloc::Allocator;

pub mod array;
pub mod cross_validation;
pub mod datasets;
pub mod ensemble;
pub mod factorization;
pub mod feature_extraction;
pub mod linear_models;
pub mod metrics;
pub mod multiclass;
pub mod svm;
pub mod traits;
pub mod trees;
pub mod utils;

#[allow(unused_imports)]
pub mod prelude {
    //! Basic data structures and traits used throughout `rustlearn`.
    pub use array::prelude::*;
    pub use traits::*;
}

use prelude::*;
use trees::decision_tree;

extern crate hyper;
use hyper::client::Client;
use std::io::{Read, Write};
use std::ffi::CStr;

extern crate libc;
use libc::c_char;

fn download_data(url: &str) -> Result<String, hyper::error::Error> {
    let client = Client::new();

    let mut output = String::new();

    let mut response = try!(client.get(url).send());
    try!(response.read_to_string(&mut output));

    Ok(output)
}

fn get_raw_data(url: &str) -> String {
    println!("Downloading data for {}", url);
    let raw_data = download_data(url).unwrap();
    println!("done.");
    raw_data
}

fn build_x_matrix_internal(url: &str) -> SparseRowArray {
    let data = get_raw_data(url);
    let mut coo = Vec::new();

    for (row, line) in data.lines().enumerate() {
        for col_str in line.split_whitespace() {
            let col = col_str.parse::<usize>().unwrap();
            coo.push((row, col));
        }
    }

    let num_rows = coo.iter().map(|x| x.0).max().unwrap() + 1;
    let num_cols = coo.iter().map(|x| x.1).max().unwrap() + 1;

    let mut array = SparseRowArray::zeros(num_rows, num_cols);

    for &(row, col) in coo.iter() {
        array.set(row, col, 1.0);
    }

    array
}

/*#[no_mangle]
pub extern "C" fn build_x_matrix(url: *const c_char) -> *mut SparseRowArray {
    let url: &str = unsafe {
        CStr::from_ptr(url).to_str().unwrap()
    };
    Box::into_raw(Box::new(build_x_matrix_internal(url)))
}*/

fn build_y_array_internal(url: &str) -> Array {
    let data = get_raw_data(url);
    let mut y = Vec::new();
    
    for line in data.lines() {
        for datum_str in line.split_whitespace() {
            let datum = datum_str.parse::<i32>().unwrap();
            y.push(datum);
        }
    }

    let array = Array::from(y.iter().map(|&x| 
                                         match x {
                                             -1 => 0.0,
                                             _ => 1.0,
                                         }
    ).collect::<Vec<f32>>());

    array
}
    
/*#[no_mangle]
pub extern "C" fn build_y_array(url: *const c_char) -> *mut Array {
    let url: &str = unsafe {
        CStr::from_ptr(url).to_str().unwrap()
    };
    Box::into_raw(Box::new(build_y_array_internal(url)))
}*/

fn get_train_data() -> (SparseRowArray, Array) {
    let X_train = build_x_matrix_internal("http://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/dorothea_train.data");
    let y_train = build_y_array_internal("http://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/dorothea_train.labels");

    (X_train, y_train)
}

#[no_mangle]
pub extern "C" fn decision_tree_train() -> *mut decision_tree::DecisionTree {
    let (X_train, y_train) = get_train_data();
    let X_train = SparseColumnArray::from(&X_train);
    
    let mut model = decision_tree::Hyperparameters::new(X_train.cols()).build();

    model.fit(&X_train, &y_train).unwrap();
    println!("Tree stats: size {}, depth {}, num nodes {}", model.get_size(),
             model.get_depth(), model.get_num_nodes());
    Box::into_raw(Box::new(model)) // copies the model into shared memory
}

#[no_mangle]
pub unsafe extern "C" fn decision_tree_predict(decision_tree: *mut decision_tree::DecisionTree,
                                               test_x_url: *const c_char,
                                               test_y_url: *const c_char) -> f32 {
    let model = decision_tree.as_ref().unwrap();
    let test_x_url: &str = unsafe {
        CStr::from_ptr(test_x_url).to_str().unwrap()
    };
    let test_y_url: &str = unsafe {
        CStr::from_ptr(test_y_url).to_str().unwrap()
    };
    let X_test = build_x_matrix_internal(test_x_url);
    let y_test = build_y_array_internal(test_y_url);
    let X_test = SparseColumnArray::from(&X_test);
    let y_test = Array::from(y_test);
    let predictions = model.predict(&X_test).unwrap();
    let accuracy = metrics::accuracy_score(&y_test, &predictions);
    accuracy
}
