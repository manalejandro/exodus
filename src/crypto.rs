//! Cryptographic primitives for the exodus protocol.
//!
//! Every hash and signature in the protocol uses a *canonical* JSON
//! serialisation so that any two nodes derive byte-identical digests for the
//! same logical record.  The canonical form matches CPython's `json.dumps`
//! (`sort_keys=True, separators=(",",":"), ensure_ascii=True`) including the
//! exact floating-point rendering, so this crate interoperates byte-for-byte
//! with the reference implementation.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand_core::OsRng;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Serialise a JSON value to the canonical (Python-compatible) string form:
/// sorted object keys, compact separators, ASCII escaping and Python float
/// formatting.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

/// Canonical bytes (UTF-8) of a JSON value.
pub fn canonical_bytes(value: &Value) -> Vec<u8> {
    canonical_json(value).into_bytes()
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                out.push_str(&py_float(f));
            }
        }
        Value::String(s) => write_escaped(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                write_escaped(key, out);
                out.push(':');
                write_value(&map[*key], out);
            }
            out.push('}');
        }
    }
}

fn write_escaped(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if (c as u32) < 0x80 => {
                out.push(c);
            }
            c if (c as u32) <= 0xffff => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => {
                // astral: UTF-16 surrogate pair escapes (matches CPython)
                let cp = c as u32 - 0x10000;
                let hi = 0xd800 + (cp >> 10);
                let lo = 0xdc00 + (cp & 0x3ff);
                out.push_str(&format!("\\u{:04x}\\u{:04x}", hi, lo));
            }
        }
    }
    out.push('"');
}

/// Render a `f64` exactly as CPython's `repr(float)`/`json.dumps` does.
///
/// Uses the shortest round-trippable digits (via `ryu`) and then applies
/// Python's notation rules: positional for `10^-4 <= v < 10^16`, otherwise
/// scientific with a signed, at-least-two-digit exponent (e.g. `1e-05`,
/// `1.5e+30`).
pub fn py_float(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() { "-Infinity".to_string() } else { "Infinity".to_string() };
    }

    let mut buffer = ryu::Buffer::new();
    let s = buffer.format_finite(v);
    let (neg, rest) = if let Some(r) = s.strip_prefix('-') { (true, r) } else { (false, s) };
    let (mantissa, sci_exp) = match rest.find('e') {
        Some(i) => (&rest[..i], rest[i + 1..].parse::<i32>().ok()),
        None => (rest, None),
    };
    let (ip, fp) = match mantissa.find('.') {
        Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
        None => (mantissa, ""),
    };
    let combined = format!("{}{}", ip, fp);
    let dec_pt = ip.len() as i32;
    let first = combined.find(|c| c != '0').unwrap_or(0) as i32;
    let mut lead_exp = dec_pt - 1 - first;
    if let Some(e) = sci_exp {
        lead_exp += e;
    }
    let sig = combined.trim_matches('0');
    let sig = if sig.is_empty() { "0" } else { sig };
    let sig_len = sig.len();

    let sign_s = if neg { "-" } else { "" };

    if lead_exp >= 16 || lead_exp < -4 {
        // scientific notation
        let mantissa = if sig_len == 1 {
            sig.to_string()
        } else {
            format!("{}.{}", &sig[..1], &sig[1..])
        };
        let esign = if lead_exp < 0 { "-" } else { "+" };
        let eabs = lead_exp.unsigned_abs();
        format!("{}{}e{}{:02}", sign_s, mantissa, esign, eabs)
    } else if lead_exp < 0 {
        // 0.0001 style: "0." then zeros then digits
        let zeros = (-lead_exp - 1) as usize;
        let mut out = String::new();
        out.push_str(sign_s);
        out.push_str("0.");
        for _ in 0..zeros {
            out.push('0');
        }
        out.push_str(sig);
        out
    } else {
        let int_digits = (lead_exp + 1) as usize;
        let mut out = String::new();
        out.push_str(sign_s);
        if int_digits >= sig_len {
            out.push_str(sig);
            for _ in 0..(int_digits - sig_len) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&sig[..int_digits]);
            out.push('.');
            out.push_str(&sig[int_digits..]);
        }
        out
    }
}

/// Hex-encoded SHA-256 digest of the canonical serialisation of a value.
pub fn sha256_hex(value: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(canonical_bytes(value));
    hex(&hasher.finalize())
}

/// Hex-encoded SHA-256 digest of raw bytes.
pub fn sha256_bytes_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex(&hasher.finalize())
}

/// Lowercase hex encoding.
pub fn hex(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for b in data {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Decode a lowercase/uppercase hex string.
pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..s.len()).step_by(2) {
        let hi = nib(bytes[i])?;
        let lo = nib(bytes[i + 1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

fn nib(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Sign *message* with the 32-byte private key, returning a 64-byte Ed25519
/// signature.
pub fn sign(message: &[u8], private_key: &[u8]) -> Vec<u8> {
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&private_key[..32]);
    let signing = SigningKey::from_bytes(&seed);
    let sig: Signature = signing.sign(message);
    sig.to_bytes().to_vec()
}

/// Verify an Ed25519 signature.  Returns `false` (never panics) on failure or
/// bad input.
pub fn verify(message: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
    let Ok(pkbuf) = <[u8; 32]>::try_from(public_key) else {
        return false;
    };
    let Ok(sigbuf) = <[u8; 64]>::try_from(signature) else {
        return false;
    };
    let Ok(pk) = VerifyingKey::from_bytes(&pkbuf) else {
        return false;
    };
    let sig = Signature::from_bytes(&sigbuf);
    pk.verify(message, &sig).is_ok()
}

/// Derive the 32-byte public key from a 32-byte private key.
pub fn public_key_from_private(private_key: &[u8]) -> Vec<u8> {
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&private_key[..32]);
    SigningKey::from_bytes(&seed).verifying_key().to_bytes().to_vec()
}

/// Generate a fresh Ed25519 key pair `(private_key, public_key)`.
pub fn generate_key_pair() -> ([u8; 32], [u8; 32]) {
    let signing = SigningKey::generate(&mut OsRng);
    (signing.to_bytes(), signing.verifying_key().to_bytes())
}

const B32: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

/// RFC 4648 base32, no padding, uppercase.
fn base32_nopad(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() * 8 + 4) / 5);
    let mut buffer: u64 = 0;
    let mut bits: i32 = 0;
    for &b in data {
        buffer = (buffer << 8) | (b as u64);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let idx = ((buffer >> bits) & 0x1f) as usize;
            out.push(B32[idx] as char);
        }
    }
    if bits > 0 {
        let idx = ((buffer << (5 - bits)) & 0x1f) as usize;
        out.push(B32[idx] as char);
    }
    out
}

/// Derive a stable node id from a public key: `"exd"` + lowercase base32 of
/// the first 16 bytes of the SHA-256 digest of the key.
pub fn node_id_from_public_key(public_key: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(public_key);
    let digest = hasher.finalize();
    let id16 = &digest[..16];
    format!("exd{}", base32_nopad(id16).to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn py_float_matches_python_repr() {
        for (val, expect) in [
            (0.0, "0.0"),
            (1.0, "1.0"),
            (-2.0, "-2.0"),
            (0.5, "0.5"),
            (0.01, "0.01"),
            (100.0, "100.0"),
            (1200.0, "1200.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e-4, "0.0001"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (2.5e-7, "2.5e-07"),
            (1.5e30, "1.5e+30"),
            (3.36e12, "3360000000000.0"),
            (123.456, "123.456"),
            (0.00015, "0.00015"),
        ] {
            assert_eq!(py_float(val), expect, "py_float({val})");
        }
    }

    #[test]
    fn canonical_sorts_keys_and_is_compact() {
        let v = json!({"b": 1, "a": 2, "c": {"z": 1, "y": 2}});
        assert_eq!(canonical_json(&v), r#"{"a":2,"b":1,"c":{"y":2,"z":1}}"#);
    }

    #[test]
    fn node_id_format() {
        let (_, pk) = generate_key_pair();
        let id = node_id_from_public_key(&pk);
        assert!(id.starts_with("exd"));
        assert_eq!(id.len(), 3 + 26);
        assert!(id[3..].chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
        assert_eq!(node_id_from_public_key(&pk), id);
    }

    #[test]
    fn sign_verify_roundtrip() {
        let (sk, pk) = generate_key_pair();
        let msg = b"hello exodus";
        let sig = sign(msg, &sk);
        assert!(verify(msg, &sig, &pk));
        assert!(!verify(b"tampered", &sig, &pk));
        assert!(!verify(msg, &sig, &[0u8; 32]));
        assert!(!verify(msg, &[], &pk));
    }
}