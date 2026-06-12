import os
import sys
from urllib.request import urlopen

import jax
import jax.numpy as jnp
from jax import Array, lax


# Bloom filter helpers — used by Go PSK detection (tiered Bloom + exact-recent cascade).
# k=4 Kirsch-Mitzenmacher: position_i = (h0 + i·h1) mod m.
# No false negatives; FP ≈ 0.9 % for m=8 192 bits and n=722 entries (19×19 Go).
_BLOOM_K: int = 4


def bloom_insert(bloom: Array, h: Array) -> Array:
    """Set the k=4 bits for 64-bit hash h in the Bloom filter.

    bloom: (W,) uint32 array — the filter with m = W·32 bits.
    h:     (2,) uint32 array — the two Zobrist hash words.
    """
    m = jnp.uint32(bloom.shape[0] * 32)
    i = jnp.arange(_BLOOM_K, dtype=jnp.uint32)
    bits    = (h[0] + i * h[1]) % m               # (k,) positions in [0, m)
    words   = (bits >> jnp.uint32(5)).astype(jnp.int32)  # word index
    offsets =  bits &  jnp.uint32(31)              # bit offset within word
    masks   = jnp.uint32(1) << offsets
    for j in range(_BLOOM_K):                      # unrolled at trace time
        bloom = bloom.at[words[j]].set(bloom[words[j]] | masks[j])
    return bloom


def bloom_query(bloom: Array, h: Array) -> Array:
    """True iff all k=4 bits for hash h are set (no false negatives; possible false positives)."""
    m = jnp.uint32(bloom.shape[0] * 32)
    i = jnp.arange(_BLOOM_K, dtype=jnp.uint32)
    bits    = (h[0] + i * h[1]) % m
    words   = (bits >> jnp.uint32(5)).astype(jnp.int32)
    offsets =  bits &  jnp.uint32(31)
    masks   = jnp.uint32(1) << offsets
    return ((bloom[words] & masks) != jnp.uint32(0)).all()


def _xor_reduce_bitparity(operand: Array, axis: int) -> Array:
    # Metal fallback: the ``bitwise_xor`` reduction primitive fails to legalize on the
    # Apple Metal (``jax-metal``) XLA backend
    # (``UNIMPLEMENTED: failed to legalize operation 'mhlo.reduce'``). Compute the XOR via
    # per-bit parity using only ``sum`` / shifts / bitwise-and. Numerically identical to the
    # native reduction, but expands each value into its bits, so it is used only on Metal.
    nbits = jnp.iinfo(operand.dtype).bits
    bitpos = jnp.arange(nbits, dtype=operand.dtype)
    bits = (jnp.expand_dims(operand, -1) >> bitpos) & 1  # (..., nbits)
    parity = jnp.sum(bits, axis=axis) & 1  # reduce the requested axis, keep bit axis
    weights = jnp.ones((), operand.dtype) << bitpos
    return jnp.sum(parity.astype(operand.dtype) * weights, axis=-1).astype(operand.dtype)


def _use_native_xor() -> bool:
    """True everywhere the native ``bitwise_xor`` reduction legalizes (CPU/CUDA/ROCm/TPU);
    False only on Apple Metal. Metal is detected by SIGNAL (a 'metal' platform or an 'apple'
    device_kind) rather than by an exact backend string, so a CUDA device (platform 'gpu',
    NVIDIA device_kind) is never misread as Metal, and a Metal device that reported platform
    'gpu' would still be caught by its Apple device_kind. Override with
    ``PGX_XOR_REDUCE=native|parity``.
    """
    forced = os.environ.get("PGX_XOR_REDUCE", "").lower()
    if forced in ("native", "parity"):
        return forced == "native"
    try:
        sig = " ".join(
            f"{getattr(d, 'platform', '')} {getattr(d, 'device_kind', '')}" for d in jax.devices()
        ).lower()
    except Exception:
        return False  # can't introspect devices -> safe portable fallback
    return not ("metal" in sig or "apple" in sig)


def xor_reduce(operand: Array, axis: int = 0) -> Array:
    """XOR-reduce an unsigned-integer array along ``axis``.

    Uses the native ``bitwise_xor`` reduction on CPU/CUDA/TPU, and falls back to a
    numerically identical per-bit-parity implementation only on the Apple Metal
    (``jax-metal``) backend, where the native reduction fails to legalize. The native path
    avoids the per-bit expansion of the fallback (~15x faster on CPU, ~100x on CUDA), which
    matters because this runs once per env step for Zobrist hashing. The backend is resolved
    at trace time, so jitted code pays no runtime cost for the check; override the choice with
    the ``PGX_XOR_REDUCE=native|parity`` environment variable.
    """
    if _use_native_xor():
        return lax.reduce(operand, jnp.zeros((), operand.dtype), lax.bitwise_xor, (axis,))
    return _xor_reduce_bitparity(operand, axis)


def _download(url, filename):
    try:
        print(f"Downloading from {url} ...", file=sys.stderr)
        data = urlopen(url).read()
        with open(filename, mode="wb") as f:
            f.write(data)
    except Exception as e:
        print(f"Failed to downalod the data from {url}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
