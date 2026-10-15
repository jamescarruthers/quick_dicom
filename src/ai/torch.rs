//! Reads the weights in a PyTorch `.pt` file, as `torch.save` writes a
//! model's state dict: a zip archive holding a pickle that names each
//! tensor and points to its raw bytes in another entry of the archive.
//!
//! Only what state dicts use is supported: the zip entries must be stored
//! uncompressed, as PyTorch writes them, and the pickle may only build
//! dicts, tuples, lists, numbers, strings and tensors.

use std::collections::HashMap;

/// A tensor of 32-bit floats, contiguous, in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Reads every floating-point tensor in a state dict, by name. Integer
/// tensors, such as batch norm's `num_batches_tracked`, are left out.
pub fn read(bytes: &[u8]) -> Result<HashMap<String, Tensor>, String> {
    let entries = zip_entries(bytes)?;
    let (pickle_name, pickle) = entries
        .iter()
        .find(|(name, _)| name.ends_with("data.pkl"))
        .ok_or("not a PyTorch file: no data.pkl")?;
    let prefix = &pickle_name[..pickle_name.len() - "data.pkl".len()];
    let root = unpickle(pickle)?;
    // A state dict, or a checkpoint that holds one under "model".
    let dict = match &root {
        Value::Dict(items) => match items
            .iter()
            .find(|(k, _)| matches!(k, Value::Str(s) if s == "model"))
        {
            Some((_, inner @ Value::Dict(_))) => inner,
            _ => &root,
        },
        _ => return Err("the file holds no state dict".into()),
    };
    let Value::Dict(items) = dict else {
        unreachable!()
    };
    let mut out = HashMap::new();
    for (key, value) in items {
        let (Value::Str(name), Value::Tensor(t)) = (key, value) else {
            continue;
        };
        let Some(size) = t.dtype.size() else {
            continue;
        };
        if !t.dtype.is_float() {
            continue;
        }
        let path = format!("{prefix}data/{}", t.key);
        let raw = entries
            .iter()
            .find(|(n, _)| *n == path)
            .ok_or_else(|| format!("{name}: missing {path}"))?
            .1;
        let numel: usize = t.shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut index = vec![0usize; t.shape.len()];
        for _ in 0..numel {
            let at = t.offset
                + index
                    .iter()
                    .zip(&t.stride)
                    .map(|(i, s)| i * s)
                    .sum::<usize>();
            let b = raw
                .get(at * size..(at + 1) * size)
                .ok_or_else(|| format!("{name}: data out of range"))?;
            data.push(t.dtype.decode(b));
            // Step the index in row-major order.
            for d in (0..index.len()).rev() {
                index[d] += 1;
                if index[d] < t.shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        out.insert(
            name.clone(),
            Tensor {
                shape: t.shape.clone(),
                data,
            },
        );
    }
    Ok(out)
}

/// The entries of a zip archive whose files are stored uncompressed.
fn zip_entries(bytes: &[u8]) -> Result<Vec<(String, &[u8])>, String> {
    let u16_at = |p: usize| {
        bytes
            .get(p..p + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
    };
    let u32_at = |p: usize| {
        bytes
            .get(p..p + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    let u64_at = |p: usize| {
        bytes
            .get(p..p + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    let bad = || "not a PyTorch file: its zip index is damaged".to_string();
    // The end of central directory record is in the last 64 KB.
    let lowest = bytes.len().saturating_sub(65_557);
    let end = (lowest..bytes.len().saturating_sub(21))
        .rev()
        .find(|&p| u32_at(p) == Some(0x0605_4b50))
        .ok_or("not a PyTorch file: no zip index")?;
    let mut count = u16_at(end + 10).ok_or_else(bad)?;
    let mut start = u32_at(end + 16).ok_or_else(bad)?;
    // Zip64: the real numbers are in the zip64 end record.
    if start == 0xFFFF_FFFF || count == 0xFFFF {
        let locator = end.checked_sub(20).ok_or_else(bad)?;
        if u32_at(locator) != Some(0x0706_4b50) {
            return Err(bad());
        }
        let record = u64_at(locator + 8).ok_or_else(bad)?;
        if u32_at(record) != Some(0x0606_4b50) {
            return Err(bad());
        }
        count = u64_at(record + 32).ok_or_else(bad)?;
        start = u64_at(record + 48).ok_or_else(bad)?;
    }
    let mut entries = Vec::with_capacity(count);
    let mut p = start;
    for _ in 0..count {
        if u32_at(p) != Some(0x0201_4b50) {
            return Err(bad());
        }
        let method = u16_at(p + 10).ok_or_else(bad)?;
        let mut size = u32_at(p + 20).ok_or_else(bad)?;
        let name_len = u16_at(p + 28).ok_or_else(bad)?;
        let extra_len = u16_at(p + 30).ok_or_else(bad)?;
        let comment_len = u16_at(p + 32).ok_or_else(bad)?;
        let mut local = u32_at(p + 42).ok_or_else(bad)?;
        let name = bytes.get(p + 46..p + 46 + name_len).ok_or_else(bad)?;
        let name = String::from_utf8_lossy(name).into_owned();
        // Zip64 extra field: the sizes and offset that did not fit, in order.
        let mut q = p + 46 + name_len;
        let extra_end = q + extra_len;
        while q + 4 <= extra_end {
            let id = u16_at(q).ok_or_else(bad)?;
            let len = u16_at(q + 2).ok_or_else(bad)?;
            if id == 1 {
                let mut r = q + 4;
                if u32_at(p + 24) == Some(0xFFFF_FFFF) {
                    r += 8; // uncompressed size, the same for stored files
                }
                if size == 0xFFFF_FFFF {
                    size = u64_at(r).ok_or_else(bad)?;
                    r += 8;
                }
                if local == 0xFFFF_FFFF {
                    local = u64_at(r).ok_or_else(bad)?;
                }
            }
            q += 4 + len;
        }
        if method != 0 {
            return Err(format!("{name} is compressed; PyTorch files are not"));
        }
        if u32_at(local) != Some(0x0403_4b50) {
            return Err(bad());
        }
        let data = local
            + 30
            + u16_at(local + 26).ok_or_else(bad)?
            + u16_at(local + 28).ok_or_else(bad)?;
        entries.push((name, bytes.get(data..data + size).ok_or_else(bad)?));
        p = extra_end + comment_len;
    }
    Ok(entries)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum DType {
    F32,
    F64,
    F16,
    BF16,
    Other(usize),
}

impl DType {
    fn from_storage(name: &str) -> DType {
        match name {
            "FloatStorage" => DType::F32,
            "DoubleStorage" => DType::F64,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "LongStorage" => DType::Other(8),
            "IntStorage" => DType::Other(4),
            "ShortStorage" => DType::Other(2),
            _ => DType::Other(1),
        }
    }

    fn size(self) -> Option<usize> {
        Some(match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 | DType::BF16 => 2,
            DType::Other(n) => n,
        })
    }

    fn is_float(self) -> bool {
        !matches!(self, DType::Other(_))
    }

    fn decode(self, b: &[u8]) -> f32 {
        match self {
            DType::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            DType::F64 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            DType::F16 => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
            DType::BF16 => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
            DType::Other(_) => 0.0,
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exponent = ((h >> 10) & 0x1F) as u32;
    let mantissa = (h & 0x3FF) as u32;
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // Subnormal: normalise it.
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((mantissa << shift) & 0x3FF) << 13
        }
        31 => sign | 0x7F80_0000 | (mantissa << 13),
        e => sign | ((e + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

#[derive(Clone, Debug)]
struct TensorRef {
    key: String,
    dtype: DType,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(String, DType),
    Tensor(TensorRef),
    /// Anything else a REDUCE built, which state dicts do not need.
    Opaque,
    Mark,
}

impl Value {
    fn int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }

    fn ints(&self) -> Option<Vec<usize>> {
        match self {
            Value::Tuple(v) | Value::List(v) => {
                v.iter().map(|x| x.int().map(|i| i as usize)).collect()
            }
            _ => None,
        }
    }
}

/// Runs the pickle machine over `data` far enough to build a state dict.
fn unpickle(data: &[u8]) -> Result<Value, String> {
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<usize, Value> = HashMap::new();
    let mut p = 0usize;
    let bad = |what: &str| format!("unreadable weights: {what}");
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let s = data
            .get(*p..*p + n)
            .ok_or("unreadable weights: truncated pickle")?;
        *p += n;
        Ok(s)
    };
    let u32_le = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap()) as usize;
    let pop_mark = |stack: &mut Vec<Value>| -> Result<Vec<Value>, String> {
        let at = stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or("unreadable weights: no mark")?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = take(&mut p, 1)?[0];
        match op {
            0x80 => {
                take(&mut p, 1)?; // PROTO
            }
            0x95 => {
                take(&mut p, 8)?; // FRAME
            }
            b'.' => break, // STOP
            b'}' => stack.push(Value::Dict(Vec::new())),
            b']' => stack.push(Value::List(Vec::new())),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b'(' => stack.push(Value::Mark),
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'K' => {
                let v = take(&mut p, 1)?[0] as i64;
                stack.push(Value::Int(v));
            }
            b'M' => {
                let b = take(&mut p, 2)?;
                stack.push(Value::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
            }
            b'J' => {
                let b = take(&mut p, 4)?;
                stack.push(Value::Int(i32::from_le_bytes(b.try_into().unwrap()) as i64));
            }
            0x8a => {
                // LONG1: little-endian two's complement.
                let n = take(&mut p, 1)?[0] as usize;
                let b = take(&mut p, n)?;
                let mut v: i64 = 0;
                for (i, &byte) in b.iter().enumerate().take(8) {
                    v |= (byte as i64) << (8 * i);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Value::Int(v));
            }
            b'G' => {
                // BINFLOAT: no state dict needs the value.
                take(&mut p, 8)?;
                stack.push(Value::Opaque);
            }
            b'X' | b'B' | b'T' => {
                let n = u32_le(take(&mut p, 4)?);
                let s = take(&mut p, n)?;
                stack.push(Value::Str(String::from_utf8_lossy(s).into_owned()));
            }
            0x8c | b'C' | b'U' => {
                let n = take(&mut p, 1)?[0] as usize;
                let s = take(&mut p, n)?;
                stack.push(Value::Str(String::from_utf8_lossy(s).into_owned()));
            }
            b'c' => {
                let line = |p: &mut usize| -> Result<String, String> {
                    let end = data[*p..]
                        .iter()
                        .position(|&b| b == b'\n')
                        .ok_or("unreadable weights: bad global")?;
                    let s = String::from_utf8_lossy(&data[*p..*p + end]).into_owned();
                    *p += end + 1;
                    Ok(s)
                };
                let module = line(&mut p)?;
                let name = line(&mut p)?;
                stack.push(Value::Global(module, name));
            }
            0x93 => {
                // STACK_GLOBAL
                let name = stack.pop().ok_or_else(|| bad("empty stack"))?;
                let module = stack.pop().ok_or_else(|| bad("empty stack"))?;
                match (module, name) {
                    (Value::Str(m), Value::Str(n)) => stack.push(Value::Global(m, n)),
                    _ => return Err(bad("bad global")),
                }
            }
            b'q' => {
                let i = take(&mut p, 1)?[0] as usize;
                memo.insert(i, stack.last().cloned().ok_or_else(|| bad("empty stack"))?);
            }
            b'r' => {
                let i = u32_le(take(&mut p, 4)?);
                memo.insert(i, stack.last().cloned().ok_or_else(|| bad("empty stack"))?);
            }
            0x94 => {
                // MEMOIZE
                let i = memo.len();
                memo.insert(i, stack.last().cloned().ok_or_else(|| bad("empty stack"))?);
            }
            b'h' => {
                let i = take(&mut p, 1)?[0] as usize;
                stack.push(memo.get(&i).cloned().ok_or_else(|| bad("bad memo"))?);
            }
            b'j' => {
                let i = u32_le(take(&mut p, 4)?);
                stack.push(memo.get(&i).cloned().ok_or_else(|| bad("bad memo"))?);
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Value::Tuple(items));
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                let at = stack
                    .len()
                    .checked_sub(n)
                    .ok_or_else(|| bad("empty stack"))?;
                let items = stack.split_off(at);
                stack.push(Value::Tuple(items));
            }
            b'a' => {
                let v = stack.pop().ok_or_else(|| bad("empty stack"))?;
                if let Some(Value::List(l)) = stack.last_mut() {
                    l.push(v);
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                if let Some(Value::List(l)) = stack.last_mut() {
                    l.extend(items);
                }
            }
            b's' => {
                let v = stack.pop().ok_or_else(|| bad("empty stack"))?;
                let k = stack.pop().ok_or_else(|| bad("empty stack"))?;
                if let Some(Value::Dict(d)) = stack.last_mut() {
                    d.push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if let Some(Value::Dict(d)) = stack.last_mut() {
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                }
            }
            b'b' => {
                // BUILD: state for the object below, such as an OrderedDict's
                // _metadata. Not needed.
                stack.pop().ok_or_else(|| bad("empty stack"))?;
            }
            b'Q' => {
                // BINPERSID: ('storage', type, key, location, size)
                let pid = stack.pop().ok_or_else(|| bad("empty stack"))?;
                let Value::Tuple(items) = pid else {
                    return Err(bad("bad storage"));
                };
                let (Some(Value::Global(_, kind)), Some(Value::Str(key))) =
                    (items.get(1), items.get(2))
                else {
                    return Err(bad("bad storage"));
                };
                stack.push(Value::Storage(key.clone(), DType::from_storage(kind)));
            }
            b'R' => {
                let args = stack.pop().ok_or_else(|| bad("empty stack"))?;
                let callable = stack.pop().ok_or_else(|| bad("empty stack"))?;
                let Value::Global(module, name) = callable else {
                    return Err(bad("call of a non-global"));
                };
                let Value::Tuple(args) = args else {
                    return Err(bad("call without a tuple"));
                };
                let value = match (module.as_str(), name.as_str()) {
                    ("collections", "OrderedDict") => Value::Dict(Vec::new()),
                    ("torch._utils", "_rebuild_tensor_v2")
                    | ("torch._utils", "_rebuild_tensor") => {
                        rebuild_tensor(&args).ok_or_else(|| bad("bad tensor"))?
                    }
                    // A tensor saved as a subclass: (rebuild function, type,
                    // its arguments, state).
                    ("torch._tensor", "_rebuild_from_type_v2") => match args.get(2) {
                        Some(Value::Tuple(inner)) => {
                            rebuild_tensor(inner).ok_or_else(|| bad("bad tensor"))?
                        }
                        _ => return Err(bad("bad tensor")),
                    },
                    // nn.Parameter(tensor, requires_grad, hooks)
                    ("torch._utils", "_rebuild_parameter")
                    | ("torch._utils", "_rebuild_parameter_with_state") => args
                        .into_iter()
                        .next()
                        .ok_or_else(|| bad("bad parameter"))?,
                    _ => Value::Opaque,
                };
                stack.push(value);
            }
            other => return Err(bad(&format!("pickle opcode 0x{other:02x}"))),
        }
    }
    stack.pop().ok_or_else(|| bad("empty pickle"))
}

/// A tensor from the arguments of `_rebuild_tensor_v2`: storage, offset,
/// shape and strides.
fn rebuild_tensor(args: &[Value]) -> Option<Value> {
    let (Some(Value::Storage(key, dtype)), Some(offset), Some(shape), Some(stride)) = (
        args.first(),
        args.get(1).and_then(Value::int),
        args.get(2).and_then(Value::ints),
        args.get(3).and_then(Value::ints),
    ) else {
        return None;
    };
    Some(Value::Tensor(TensorRef {
        key: key.clone(),
        dtype: *dtype,
        offset: offset as usize,
        shape,
        stride,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file `torch.save` wrote: tensors in float32 (one transposed, one a
    /// slice of a larger storage), float16 and int64.
    #[test]
    fn reads_a_state_dict() {
        let tensors = read(include_bytes!("testdata/tiny.pt")).unwrap();
        let a = &tensors["a.weight"];
        assert_eq!(a.shape, [2, 3, 4]);
        assert_eq!(a.data, (0..24).map(|i| i as f32).collect::<Vec<_>>());
        // arange(6).reshape(2, 3).t()
        assert_eq!(tensors["b.t"].shape, [3, 2]);
        assert_eq!(tensors["b.t"].data, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(tensors["d.half"].data, [1.5, -2.25]);
        assert_eq!(tensors["e.slice"].data, [3.0, 4.0, 5.0, 6.0]);
        assert!(!tensors.contains_key("c.num_batches_tracked"));
    }

    #[test]
    fn half_floats_decode() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        assert!(read(b"not a zip at all").is_err());
        assert!(read(&[0u8; 100]).is_err());
    }
}
