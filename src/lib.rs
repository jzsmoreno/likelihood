use pyo3::prelude::*;

#[pyfunction]
fn print_hello() {
    println!("Hello from Rust integration!");
}

#[pymodule]
fn rust_py_integration(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_hello, m)?)?;
    Ok(())
}
