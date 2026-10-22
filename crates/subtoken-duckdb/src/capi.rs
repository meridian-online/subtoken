//! A small safe layer over DuckDB 2.0's v2 C API: just what the seven scalar
//! functions need to register themselves with their docs, read a VARCHAR
//! argument, and write their results.
//!
//! Every v2 call returns a `DUCKDB_V2_ERROR` and takes an error slot. [`v2!`]
//! passes a slot of our own, initialised to NULL; on failure DuckDB allocates an
//! error info into it, which we read and then destroy, since a slot we pass is
//! ours to clean up. The only slots we do not own are the ones DuckDB hands to a
//! callback or to the entrypoint: those are written with
//! `error_info_set_code`/`set_text` and never destroyed.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe, UnwindSafe};
use std::ptr;
use std::sync::Arc;

use subtoken_duckdb_sys::*;

/// Calls a v2 API function with a fresh error slot appended, turning a failure
/// into `Err` with DuckDB's text. Must be used inside an `unsafe` block in a
/// function returning `Result<_, String>`.
macro_rules! v2 {
    ($f:ident($($arg:expr),* $(,)?)) => {{
        let f = api()
            .$f
            .ok_or(concat!(stringify!($f), " is missing from DuckDB's API table"))?;
        let mut err: duckdb_v2_error_info_handle = ptr::null_mut();
        let code = f($($arg,)* &mut err);
        check(code, err, stringify!($f))
    }};
}

/// Turns a call's return code into a `Result`, consuming the error info DuckDB
/// wrote into our slot on failure.
///
/// # Safety
///
/// `err` is NULL or an error info DuckDB allocated into a slot we passed.
unsafe fn check(
    code: DUCKDB_V2_ERROR,
    mut err: duckdb_v2_error_info_handle,
    call: &str,
) -> Result<(), String> {
    let result = if code == DUCKDB_V2_ERROR_DUCKDB_V2_ERROR_NONE {
        Ok(())
    } else {
        // SAFETY: `err` is NULL or a live info, per this function's contract.
        Err(unsafe { error_text(err) }
            .unwrap_or_else(|| format!("{call} failed with DuckDB error code {code}")))
    };
    if !err.is_null() {
        if let Some(destroy) = api().duckdb_v2_error_info_destroy {
            // SAFETY: the info was allocated into our slot, so it is ours to free.
            unsafe { destroy(&mut err) };
        }
    }
    result
}

/// The text of an error info, or `None` if there is none to read.
///
/// # Safety
///
/// `err` is NULL or a live error info.
unsafe fn error_text(err: duckdb_v2_error_info_handle) -> Option<String> {
    if err.is_null() {
        return None;
    }
    let get_text = api().duckdb_v2_error_info_get_text?;
    let mut text = duckdb_v2_str {
        ptr: ptr::null(),
        len: 0,
    };
    // SAFETY: `err` is live; `text` receives a view borrowed from it.
    let code = unsafe { get_text(err, &mut text) };
    if code != DUCKDB_V2_ERROR_DUCKDB_V2_ERROR_NONE || text.ptr.is_null() || text.len == 0 {
        return None;
    }
    // SAFETY: DuckDB's view points at `len` bytes that live as long as `err`.
    let bytes = unsafe { std::slice::from_raw_parts(text.ptr.cast::<u8>(), text.len as usize) };
    Some(String::from_utf8_lossy(bytes).into_owned())
}

/// A borrowed view of `text`, valid while `text` is.
fn view(text: &str) -> duckdb_v2_str {
    duckdb_v2_str {
        ptr: text.as_ptr().cast(),
        len: text.len() as idx_t,
    }
}

/// Runs `f`, turning a panic into `Err("subtoken panicked: <message>")`, so
/// nothing unwinds across an `extern "C"` boundary.
pub fn guard<T>(f: impl FnOnce() -> Result<T, String> + UnwindSafe) -> Result<T, String> {
    match catch_unwind(f) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("a panic with no message");
            Err(format!("subtoken panicked: {message}"))
        }
    }
}

/// What `duckdb_functions()` reports for a function.
pub struct Docs {
    pub description: &'static str,
    pub example: &'static str,
    pub category: &'static str,
}

/// What registration needs from the entrypoint's input: the extension to
/// register on, and a factory to create functions and types through. Both are
/// borrowed from the input and valid only until the entrypoint returns.
pub struct Registrar {
    extension: duckdb_v2_extension_handle,
    factory: duckdb_v2_factory_handle,
}

impl Registrar {
    /// Takes the extension from `input` and the factory from `input.context`.
    ///
    /// # Safety
    ///
    /// `input` is the live input DuckDB passed to the entrypoint,
    /// `subtoken_duckdb_sys::init` has succeeded on it, and the `Registrar` is
    /// dropped before the entrypoint returns.
    pub unsafe fn from_input(input: *mut duckdb_v2_extension_input) -> Result<Self, String> {
        // SAFETY: the caller guarantees `input` is live.
        let input = unsafe { &*input };
        let mut factory: duckdb_v2_factory_handle = ptr::null_mut();
        // SAFETY: `input.context` is the entrypoint's live context.
        unsafe { v2!(duckdb_v2_context_get_factory(input.context, &mut factory)) }?;
        Ok(Self {
            extension: input.extension,
            factory,
        })
    }

    /// Parses SQL type text into an owned type.
    fn create_type(&self, text: &str) -> Result<OwnedType, String> {
        let mut handle: duckdb_v2_logical_type_handle = ptr::null_mut();
        // SAFETY: `factory` is live for the Registrar's life (see `from_input`).
        unsafe {
            v2!(duckdb_v2_factory_create_type_from_text(
                self.factory,
                &view(text),
                &mut handle
            ))
        }?;
        Ok(OwnedType(handle))
    }
}

/// A logical type we created, destroyed on drop.
struct OwnedType(duckdb_v2_logical_type_handle);

impl Drop for OwnedType {
    fn drop(&mut self) {
        if let Some(destroy) = api().duckdb_v2_logical_type_destroy {
            // SAFETY: the type came from `create_type_from_text` and is ours.
            unsafe { destroy(&mut self.0) };
        }
    }
}

/// A scalar function's body: read the arguments from `Call` and write its
/// result there. An `Err` is raised by DuckDB as the query's error.
pub type Exec = fn(&mut Call) -> Result<(), String>;

/// A scalar function to register: name, parameters, return type, docs.
pub struct ScalarFunction {
    name: &'static str,
    parameters: Vec<(&'static str, &'static str)>,
    returns: Option<String>,
    docs: Option<&'static Docs>,
    volatile: bool,
    exec: Option<Exec>,
}

impl ScalarFunction {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            parameters: Vec::new(),
            returns: None,
            docs: None,
            volatile: false,
            exec: None,
        }
    }

    /// Adds a STANDARD parameter (passable by position or by name) of the SQL
    /// type `ty`.
    pub fn parameter(mut self, name: &'static str, ty: &'static str) -> Self {
        self.parameters.push((name, ty));
        self
    }

    /// Sets the return type, as SQL type text.
    pub fn returns(mut self, ty: impl Into<String>) -> Self {
        self.returns = Some(ty.into());
        self
    }

    pub fn docs(mut self, docs: &'static Docs) -> Self {
        self.docs = Some(docs);
        self
    }

    /// Marks the function VOLATILE, so DuckDB calls it every time rather than
    /// folding a zero-argument call to a constant.
    pub fn volatile(mut self) -> Self {
        self.volatile = true;
        self
    }

    /// Sets the function's body.
    ///
    /// `exec` reads and writes through [`Call`], whose readers and writers
    /// trust that each argument and the result have the SQL types this
    /// builder registered: `varchar_arg` reads a VARCHAR, `write_float_lists`
    /// writes a `FLOAT[]`, and so on. Nothing at run time checks the pairing,
    /// so calling a reader or writer for a type the function was not
    /// registered with reads or writes memory of the wrong layout.
    pub fn exec(mut self, exec: Exec) -> Self {
        self.exec = Some(exec);
        self
    }

    /// Creates, configures and registers the function on the extension, then
    /// destroys the handle, which the API requires after registration too.
    pub fn register(&self, registrar: &Registrar) -> Result<(), String> {
        let exec = self
            .exec
            .ok_or_else(|| format!("{} has no exec callback", self.name))?;
        let returns = self
            .returns
            .as_deref()
            .ok_or_else(|| format!("{} has no return type", self.name))?;
        let mut function: duckdb_v2_scalar_function_handle = ptr::null_mut();
        // SAFETY: the factory is live for the Registrar's life.
        unsafe {
            v2!(duckdb_v2_scalar_function_create(
                registrar.factory,
                &mut function
            ))
        }?;
        let result = self.configure_and_register(registrar, function, returns, exec);
        if let Some(destroy) = api().duckdb_v2_scalar_function_destroy {
            // SAFETY: `function` was created above and is ours to destroy.
            unsafe { destroy(&mut function) };
        }
        result.map_err(|e| format!("registering {}: {e}", self.name))
    }

    fn configure_and_register(
        &self,
        registrar: &Registrar,
        function: duckdb_v2_scalar_function_handle,
        returns: &str,
        exec: Exec,
    ) -> Result<(), String> {
        // SAFETY (whole body): `function` is a live handle we created, the
        // signature and docs handles are borrowed from it, every view points at
        // a `&str` that outlives the call, and the types live until the calls
        // that copy them return.
        unsafe {
            v2!(duckdb_v2_scalar_function_set_name(
                function,
                &view(self.name)
            ))?;

            let mut signature: duckdb_v2_function_signature_handle = ptr::null_mut();
            v2!(duckdb_v2_scalar_function_get_signature(
                function,
                &mut signature
            ))?;
            for (name, ty) in &self.parameters {
                let ty = registrar.create_type(ty)?;
                v2!(duckdb_v2_function_signature_add_parameter(
                    signature,
                    &view(name),
                    ty.0,
                    ptr::null_mut(),
                    DUCKDB_V2_FUNCTION_PARAMETER_KIND_DUCKDB_V2_FUNCTION_PARAMETER_KIND_STANDARD,
                ))?;
            }
            let return_type = registrar.create_type(returns)?;
            v2!(duckdb_v2_function_signature_set_return_type(
                signature,
                return_type.0
            ))?;

            if let Some(docs) = self.docs {
                let mut handle: duckdb_v2_function_docs_handle = ptr::null_mut();
                v2!(duckdb_v2_scalar_function_get_docs(function, &mut handle))?;
                v2!(duckdb_v2_function_docs_set_description(
                    handle,
                    &view(docs.description)
                ))?;
                v2!(duckdb_v2_function_docs_add_example(
                    handle,
                    ptr::null(),
                    &view(docs.example)
                ))?;
                v2!(duckdb_v2_function_docs_add_category(
                    handle,
                    &view(docs.category)
                ))?;
            }

            if self.volatile {
                v2!(duckdb_v2_scalar_function_set_property(
                    function,
                    DUCKDB_V2_FUNCTION_PROPERTY_KEY_DUCKDB_V2_FUNCTION_PROPERTY_STABILITY,
                    DUCKDB_V2_FUNCTION_PROPERTY_VALUE_DUCKDB_V2_FUNCTION_PROPERTY_STABILITY_VOLATILE,
                ))?;
            }

            // The user data carries this function's `Exec` to the shared
            // trampoline. DuckDB copies the opaque into a shared owner and calls
            // `drop_exec` when the last function referring to it is gone. If the
            // call fails it has not taken ownership, so the box is freed here.
            let exec_ptr = Box::into_raw(Box::new(exec));
            let mut opaque = duckdb_v2_opaque {
                ptr: exec_ptr.cast(),
                destroy: Some(drop_exec),
                equals: None,
            };
            if let Err(e) = v2!(duckdb_v2_scalar_function_set_user_data(
                function,
                &mut opaque
            )) {
                drop(Box::from_raw(exec_ptr));
                return Err(e);
            }
            v2!(duckdb_v2_scalar_function_set_exec_callback(
                function,
                Some(exec_trampoline)
            ))?;
            v2!(duckdb_v2_extension_register_scalar_function(
                registrar.extension,
                function
            ))
        }
    }
}

/// Frees the boxed `Exec` a function's user data points at.
unsafe extern "C" fn drop_exec(data: *mut c_void) {
    // SAFETY: `data` is the `Box<Exec>` leaked in `configure_and_register`, and
    // DuckDB calls this once. Nothing here should panic, but no unwind may cross
    // the boundary, so the body is guarded anyway.
    let _ = catch_unwind(|| drop(unsafe { Box::from_raw(data.cast::<Exec>()) }));
}

/// The exec callback every registered function shares: it finds the
/// function's `Exec` in the user data and runs it inside [`guard`]. A panic or
/// `Err` is written into DuckDB's error slot, which DuckDB raises after the
/// callback returns.
unsafe extern "C" fn exec_trampoline(
    info: duckdb_v2_scalar_function_exec_info_handle,
    _context: duckdb_v2_context_handle,
    err: *mut duckdb_v2_error_info_handle,
) {
    let result = guard(AssertUnwindSafe(|| {
        let mut data: *mut c_void = ptr::null_mut();
        // SAFETY: `info` is the live exec info DuckDB passed in.
        unsafe {
            v2!(duckdb_v2_scalar_function_exec_get_user_data(
                info, &mut data
            ))
        }?;
        if data.is_null() {
            return Err("subtoken: the function has no exec callback attached".to_string());
        }
        // SAFETY: the user data is the `Box<Exec>` set at registration, and it
        // lives as long as the registered function.
        let exec = unsafe { *data.cast::<Exec>() };
        // SAFETY: `info` is live for this callback.
        let mut call = unsafe { Call::new(info) }?;
        exec(&mut call)
    }));
    if let Err(message) = result {
        // `report` calls `api()`, which panics if `init` never ran, so it is
        // guarded too; if it panics there is nothing left to report through.
        // SAFETY: DuckDB passes a live slot pointing at a live info.
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe { report(err, &message) }));
    }
}

/// Writes an error into a slot DuckDB owns. The slot is borrowed, never
/// destroyed.
///
/// # Safety
///
/// `err` is NULL or a slot DuckDB handed to a callback or to the entrypoint.
pub unsafe fn report(err: *mut duckdb_v2_error_info_handle, message: &str) {
    if err.is_null() {
        return;
    }
    // SAFETY: the caller guarantees `err` is a live slot.
    let info = unsafe { *err };
    let api = api();
    if let (Some(set_code), Some(set_text)) = (
        api.duckdb_v2_error_info_set_code,
        api.duckdb_v2_error_info_set_text,
    ) {
        // SAFETY: `info` is DuckDB's live info; set_text copies the view, and
        // both accept a NULL info as a no-op.
        unsafe {
            set_code(info, DUCKDB_V2_ERROR_DUCKDB_V2_ERROR_INPUT_INVALID);
            set_text(info, &view(message));
        }
    }
}

/// One invocation of a scalar function: its input rows and its result vector.
/// Valid only inside the exec callback.
///
/// # The type contract
///
/// The readers and writers here trust that the column or result they touch
/// has the type the function was registered with (see
/// [`ScalarFunction::exec`]): `varchar_arg(i)` is only for a VARCHAR
/// parameter, `write_float_lists` only for a `FLOAT[]` result,
/// `write_bools` for BOOLEAN, `write_bigints` for BIGINT,
/// `write_varchar_constant` for VARCHAR, and `struct_child` for a STRUCT
/// whose field `i` has the type the [`Child`] writer assumes. Nothing checks
/// this at run time, so a mismatch is undefined behaviour rather than an
/// error.
///
/// The writers also assume the result vector is FLAT, which is what DuckDB's
/// expression executor hands a scalar function. [`Call::new`] checks that once
/// per call and refuses anything else with an error.
pub struct Call {
    info: duckdb_v2_scalar_function_exec_info_handle,
    rows: usize,
    result: duckdb_v2_vector_handle,
}

impl Call {
    /// # Safety
    ///
    /// `info` is the live exec info of the current callback.
    unsafe fn new(info: duckdb_v2_scalar_function_exec_info_handle) -> Result<Self, String> {
        let mut rows: idx_t = 0;
        let mut result: duckdb_v2_vector_handle = ptr::null_mut();
        // SAFETY: `info` is live, per this function's contract.
        unsafe {
            v2!(duckdb_v2_scalar_function_exec_get_row_count(
                info, &mut rows
            ))?;
            v2!(duckdb_v2_scalar_function_exec_get_result(info, &mut result))?;
            let mut kind: DUCKDB_V2_VECTOR_TYPE = DUCKDB_V2_VECTOR_TYPE_DUCKDB_V2_VECTOR_TYPE_OTHER;
            v2!(duckdb_v2_vector_get_vector_type(result, &mut kind))?;
            if kind != DUCKDB_V2_VECTOR_TYPE_DUCKDB_V2_VECTOR_TYPE_FLAT {
                return Err(format!(
                    "subtoken expected DuckDB to hand it a FLAT result vector, and got vector type {kind}"
                ));
            }
        }
        Ok(Self {
            info,
            rows: rows as usize,
            result,
        })
    }

    /// How many rows this call must produce.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Reads argument `col`, a VARCHAR, one entry per row, `None` for NULL.
    ///
    /// CONSTANT and DICTIONARY vectors are read through their selection vector;
    /// a vector the view refuses (FSST, SEQUENCE, shredded) is flattened first.
    pub fn varchar_arg(&mut self, col: usize) -> Result<Vec<Option<String>>, String> {
        let mut vector: duckdb_v2_vector_handle = ptr::null_mut();
        let mut kind: DUCKDB_V2_VECTOR_TYPE = DUCKDB_V2_VECTOR_TYPE_DUCKDB_V2_VECTOR_TYPE_OTHER;
        let mut v = duckdb_v2_vector_view {
            data: ptr::null(),
            validity: ptr::null(),
            sel: ptr::null(),
            count: 0,
        };
        // SAFETY (whole block): `info` is live for the callback, the argument
        // vector is borrowed from it, and the view's pointers stay valid until
        // the chunk is destroyed, after this function returns. The column is
        // VARCHAR by the registered signature, so `data` holds `bytes`; a
        // physical index is `sel[i]`, or `i` without a selection vector, and
        // is within the vector for every `i < rows`.
        unsafe {
            v2!(duckdb_v2_scalar_function_exec_get_arg(
                self.info,
                col as idx_t,
                &mut vector
            ))?;
            v2!(duckdb_v2_vector_get_vector_type(vector, &mut kind))?;
            if kind == DUCKDB_V2_VECTOR_TYPE_DUCKDB_V2_VECTOR_TYPE_OTHER {
                v2!(duckdb_v2_vector_flatten(vector))?;
            }
            v2!(duckdb_v2_vector_get_view(vector, &mut v))?;
            let data = v.data.cast::<duckdb_v2_bytes>();
            let mut out = Vec::with_capacity(self.rows);
            for i in 0..self.rows {
                let idx = if v.sel.is_null() {
                    i
                } else {
                    *v.sel.add(i) as usize
                };
                let valid =
                    v.validity.is_null() || (*v.validity.add(idx / 64) >> (idx % 64)) & 1 == 1;
                if valid {
                    let bytes = decode_bytes(&*data.add(idx));
                    out.push(Some(String::from_utf8_lossy(bytes).into_owned()));
                } else {
                    out.push(None);
                }
            }
            Ok(out)
        }
    }

    /// Writes one `FLOAT[]` per row; a `None` row is NULL with an
    /// `(offset, 0)` entry.
    pub fn write_float_lists(&mut self, rows: &[Option<Arc<[f32]>>]) -> Result<(), String> {
        self.check_len(rows.len())?;
        let lengths: Vec<Option<usize>> = rows
            .iter()
            .map(|row| row.as_ref().map(|v| v.len()))
            .collect();
        let (entries, total) = list_entries(&lengths);
        let mut child: duckdb_v2_vector_handle = ptr::null_mut();
        // SAFETY (whole block): the result vector is live for the callback and
        // is FLOAT[] by the registered return type. The child is sized before
        // its data pointer is fetched, because reserving can move the data;
        // `total` floats and `rows.len()` entries are then in bounds.
        unsafe {
            v2!(duckdb_v2_vector_set_size(self.result, rows.len() as idx_t))?;
            v2!(duckdb_v2_vector_get_child(self.result, 0, &mut child))?;
            v2!(duckdb_v2_vector_set_size(child, total as idx_t))?;
            if total > 0 {
                let floats = data_mutable(child)?.cast::<f32>();
                let mut offset = 0;
                for vector in rows.iter().flatten() {
                    ptr::copy_nonoverlapping(vector.as_ptr(), floats.add(offset), vector.len());
                    offset += vector.len();
                }
            }
            let slots = data_mutable(self.result)?.cast::<duckdb_v2_list_entry>();
            ptr::copy_nonoverlapping(entries.as_ptr(), slots, entries.len());
            for (row, value) in rows.iter().enumerate() {
                if value.is_none() {
                    v2!(duckdb_v2_vector_set_null(self.result, row as idx_t))?;
                }
            }
        }
        Ok(())
    }

    /// Writes one BOOLEAN per row, `None` as NULL.
    pub fn write_bools(&mut self, rows: &[Option<bool>]) -> Result<(), String> {
        self.check_len(rows.len())?;
        // SAFETY: the result vector is live and BOOLEAN by the registered
        // return type; it is sized to `rows.len()` before the data is fetched.
        unsafe {
            v2!(duckdb_v2_vector_set_size(self.result, rows.len() as idx_t))?;
            let data = data_mutable(self.result)?.cast::<bool>();
            for (row, value) in rows.iter().enumerate() {
                *data.add(row) = value.unwrap_or(false);
            }
            for (row, value) in rows.iter().enumerate() {
                if value.is_none() {
                    v2!(duckdb_v2_vector_set_null(self.result, row as idx_t))?;
                }
            }
        }
        Ok(())
    }

    /// Writes one BIGINT per row.
    pub fn write_bigints(&mut self, rows: &[i64]) -> Result<(), String> {
        self.check_len(rows.len())?;
        // SAFETY: the result vector is live and BIGINT by the registered
        // return type.
        unsafe { write_i64s(self.result, rows) }
    }

    /// Writes `text` into every row of a VARCHAR result.
    pub fn write_varchar_constant(&mut self, text: &str) -> Result<(), String> {
        // SAFETY: the result vector is live and VARCHAR by the registered
        // return type.
        unsafe { write_varchar_rows(self.result, self.rows, text) }
    }

    /// Field `i` of a STRUCT result, sized to this call's rows.
    pub fn struct_child(&mut self, i: usize) -> Result<Child<'_>, String> {
        let mut vector: duckdb_v2_vector_handle = ptr::null_mut();
        // SAFETY: the result vector is live and a STRUCT by the registered
        // return type; sizing a STRUCT sizes its fields.
        unsafe {
            v2!(duckdb_v2_vector_set_size(self.result, self.rows as idx_t))?;
            v2!(duckdb_v2_vector_get_child(
                self.result,
                i as idx_t,
                &mut vector
            ))?;
        }
        Ok(Child {
            vector,
            rows: self.rows,
            _call: PhantomData,
        })
    }

    fn check_len(&self, len: usize) -> Result<(), String> {
        if len == self.rows {
            Ok(())
        } else {
            Err(format!(
                "subtoken wrote {len} rows where DuckDB asked for {}",
                self.rows
            ))
        }
    }
}

/// One field of a STRUCT result, borrowed from its [`Call`].
pub struct Child<'a> {
    vector: duckdb_v2_vector_handle,
    rows: usize,
    _call: PhantomData<&'a mut Call>,
}

impl Child<'_> {
    /// Writes `text` into every row of a VARCHAR field.
    pub fn write_varchar_constant(&mut self, text: &str) -> Result<(), String> {
        // SAFETY: the field is live for the callback and VARCHAR by the
        // registered return type.
        unsafe { write_varchar_rows(self.vector, self.rows, text) }
    }

    /// Writes `value` into every row of a BIGINT field.
    pub fn write_bigint_constant(&mut self, value: i64) -> Result<(), String> {
        // SAFETY: the field is live for the callback and BIGINT by the
        // registered return type.
        unsafe { write_i64s(self.vector, &vec![value; self.rows]) }
    }
}

/// # Safety
///
/// `vector` is a live FLAT or CONSTANT vector.
unsafe fn data_mutable(vector: duckdb_v2_vector_handle) -> Result<*mut c_void, String> {
    let mut data: *mut c_void = ptr::null_mut();
    // SAFETY: per this function's contract.
    unsafe { v2!(duckdb_v2_vector_get_data_mutable(vector, &mut data)) }?;
    Ok(data)
}

/// # Safety
///
/// `vector` is a live, writable BIGINT vector.
unsafe fn write_i64s(vector: duckdb_v2_vector_handle, rows: &[i64]) -> Result<(), String> {
    // SAFETY: sized to `rows.len()` before the data is fetched, so the copy is
    // in bounds.
    unsafe {
        v2!(duckdb_v2_vector_set_size(vector, rows.len() as idx_t))?;
        let data = data_mutable(vector)?.cast::<i64>();
        ptr::copy_nonoverlapping(rows.as_ptr(), data, rows.len());
    }
    Ok(())
}

/// Writes `text` into `rows` rows. A string longer than the inline limit is
/// copied into the vector's arena once, and every row points at that copy.
///
/// # Safety
///
/// `vector` is a live, writable VARCHAR vector.
unsafe fn write_varchar_rows(
    vector: duckdb_v2_vector_handle,
    rows: usize,
    text: &str,
) -> Result<(), String> {
    // SAFETY (whole block): sized to `rows` before the data is fetched; the
    // arena allocation is `text.len()` bytes and lives as long as the vector.
    unsafe {
        v2!(duckdb_v2_vector_set_size(vector, rows as idx_t))?;
        let mut heap: *mut u8 = ptr::null_mut();
        if text.len() > DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
            let mut arena: duckdb_v2_arena_handle = ptr::null_mut();
            v2!(duckdb_v2_vector_get_arena(vector, &mut arena))?;
            v2!(duckdb_v2_arena_allocate(
                arena,
                text.len() as idx_t,
                &mut heap
            ))?;
        }
        let value = encode_bytes(text.as_bytes(), |_| heap);
        let data = data_mutable(vector)?.cast::<duckdb_v2_bytes>();
        for row in 0..rows {
            *data.add(row) = value;
        }
    }
    Ok(())
}

/// Encodes `s` as DuckDB's 16-byte string storage: inline up to
/// `DUCKDB_V2_BYTES_INLINE_LENGTH` bytes, zero-padded; otherwise copied to the
/// buffer `arena(s.len())` returns, with the first four bytes as the prefix.
///
/// # Safety
///
/// For a value longer than the inline limit, `arena` returns a pointer to at
/// least `s.len()` writable bytes, which must outlive every read of the result.
///
/// # Panics
///
/// If `s` is longer than `u32::MAX` bytes, which the storage cannot express.
unsafe fn encode_bytes(s: &[u8], arena: impl FnOnce(usize) -> *mut u8) -> duckdb_v2_bytes {
    let length = u32::try_from(s.len()).expect("a string longer than 4 GiB");
    // SAFETY: all-zero is a valid `bytes`: an empty inline value. Inline
    // values must be zero-padded, because DuckDB compares all 16 bytes.
    let mut b: duckdb_v2_bytes = unsafe { std::mem::zeroed() };
    if s.len() <= DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
        let mut inlined = [0 as core::ffi::c_char; 12];
        for (slot, byte) in inlined.iter_mut().zip(s) {
            *slot = *byte as core::ffi::c_char;
        }
        b.value.inlined = duckdb_v2_bytes__bindgen_ty_1__bindgen_ty_2 { length, inlined };
    } else {
        let heap = arena(s.len());
        // SAFETY: the caller guarantees `heap` holds `s.len()` bytes.
        unsafe { ptr::copy_nonoverlapping(s.as_ptr(), heap, s.len()) };
        let mut prefix = [0 as core::ffi::c_char; 4];
        for (slot, byte) in prefix.iter_mut().zip(s) {
            *slot = *byte as core::ffi::c_char;
        }
        b.value.pointer = duckdb_v2_bytes__bindgen_ty_1__bindgen_ty_1 {
            length,
            prefix,
            ptr: heap.cast(),
        };
    }
    b
}

/// The bytes a 16-byte string value holds.
///
/// # Safety
///
/// A value longer than the inline limit points at `length` live bytes, which
/// outlive the returned slice.
unsafe fn decode_bytes(b: &duckdb_v2_bytes) -> &[u8] {
    // SAFETY: both union arms start with the length.
    let length = unsafe { b.value.inlined.length } as usize;
    if length <= DUCKDB_V2_BYTES_INLINE_LENGTH as usize {
        // SAFETY: a short value is stored inline, in the value itself.
        let inlined = unsafe { &b.value.inlined.inlined };
        // SAFETY: `c_char` and `u8` have the same layout.
        unsafe { std::slice::from_raw_parts(inlined.as_ptr().cast::<u8>(), length) }
    } else {
        // SAFETY: per this function's contract.
        unsafe { std::slice::from_raw_parts(b.value.pointer.ptr.cast::<u8>(), length) }
    }
}

/// The `{offset, length}` entries of a LIST column with these row lengths, and
/// the total child length. A NULL row gets `(offset, 0)`, so a reader that
/// ignores validity still sees a well-formed entry.
fn list_entries(lengths: &[Option<usize>]) -> (Vec<duckdb_v2_list_entry>, usize) {
    let mut offset = 0usize;
    let entries = lengths
        .iter()
        .map(|length| {
            let length = length.unwrap_or(0);
            let entry = duckdb_v2_list_entry {
                offset: offset as idx_t,
                length: length as idx_t,
            };
            offset += length;
            entry
        })
        .collect();
    (entries, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twelve_bytes_are_stored_inline() {
        // SAFETY: an inline value never touches the arena, and decode reads the
        // value just encoded.
        let b = unsafe { encode_bytes(b"abcdefghijkl", |_| panic!("arena used")) };
        assert_eq!(unsafe { decode_bytes(&b) }, b"abcdefghijkl");
    }

    #[test]
    fn thirteen_bytes_go_to_the_arena() {
        let mut arena = vec![0u8; 13];
        let mut calls = Vec::new();
        // SAFETY: `arena` holds 13 bytes and outlives `b`.
        let b = unsafe {
            encode_bytes(b"abcdefghijklm", |len| {
                calls.push(len);
                arena.as_mut_ptr()
            })
        };
        assert_eq!(calls, [13]);
        // SAFETY: `b` points into `arena`, which is still alive.
        assert_eq!(unsafe { decode_bytes(&b) }, b"abcdefghijklm");
        // SAFETY: a 13-byte value is stored in the pointer form.
        let prefix = unsafe { b.value.pointer.prefix };
        assert_eq!(prefix.map(|c| c as u8), *b"abcd");
    }

    #[test]
    fn multibyte_text_round_trips() {
        let text = "한국어 텍스트 🙂".as_bytes();
        let mut arena = vec![0u8; text.len()];
        // SAFETY: `arena` holds `text.len()` bytes and outlives `b`.
        let b = unsafe { encode_bytes(text, |_| arena.as_mut_ptr()) };
        // SAFETY: `b` points into `arena`, which is still alive.
        assert_eq!(unsafe { decode_bytes(&b) }, text);
    }

    #[test]
    fn null_rows_keep_a_well_formed_offset() {
        let (entries, total) = list_entries(&[Some(3), None, Some(2)]);
        let pairs: Vec<(u64, u64)> = entries.iter().map(|e| (e.offset, e.length)).collect();
        assert_eq!(pairs, [(0, 3), (3, 0), (3, 2)]);
        assert_eq!(total, 5);
    }

    #[test]
    fn a_panic_becomes_an_error() {
        let result = guard(|| -> Result<(), String> { panic!("boom") });
        let err = result.unwrap_err();
        assert!(err.contains("boom"), "{err}");
    }
}
