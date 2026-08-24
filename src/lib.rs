use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;


#[pyfunction]
#[pyo3(signature = (data, n_cols, similarity, threshold, sparse=true))]
fn cal_adjacency_matrix(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    n_cols: usize,
    similarity: usize,
    threshold: f64,
    sparse: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    if n_cols == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be greater than zero",
        ));
    }

    let data_array = data.as_array();

    let n = data_array.shape()[0];

    if data_array.shape()[1] != n_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols does not match the input array",
        ));
    }

    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Data must contain at least one row",
        ));
    }

    if similarity > n_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "similarity cannot be greater than n_cols",
        ));
    }

    // `sparse` is accepted for API compatibility.
    // This implementation currently returns a dense matrix.
    let _ = sparse;

    let edges: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            ((i + 1)..n).filter_map(move |j| {
                let mut match_count = 0;

                for k in 0..n_cols {
                    let a = data_array[[i, k]];
                    let b = data_array[[j, k]];

                    if a.is_finite() && b.is_finite() {
                        if a == 0.0 && b == 0.0 {
                            match_count += 1;
                        } else if a != 0.0 && b != 0.0 {
                            let ratio = a.max(b) / a.min(b);

                            if ratio >= 1.0 - threshold
                                && ratio <= 1.0 + threshold
                            {
                                match_count += 1;
                            }
                        }
                    }

                    if match_count >= similarity {
                        break;
                    }
                }

                if match_count >= similarity {
                    Some((i, j))
                } else {
                    None
                }
            })
        })
        .collect();

    let adjacency = PyArray2::<f64>::zeros(py, [n, n], false);

    for (i, j) in edges {
        unsafe {
            *adjacency.uget_mut([i, j]) = 1.0;
            *adjacency.uget_mut([j, i]) = 1.0;
        }
    }

    Ok(adjacency.unbind())
}


#[pyfunction]
fn print_hello() {
    println!("Hello from Rust!");
}


#[pymodule]
fn rust_py_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_hello, m)?)?;
    m.add_function(wrap_pyfunction!(cal_adjacency_matrix, m)?)?;

    Ok(())
}
