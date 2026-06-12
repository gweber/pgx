import sys
from urllib.request import urlopen

import jax.numpy as jnp
from jax import Array


# Bloom filter helpers — used by Go PSK detection.
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


def xor_reduce(operand: Array, axis: int = 0) -> Array:
    """XOR-reduce an unsigned-integer array along ``axis`` via per-bit parity.

    Equivalent to ``lax.reduce(operand, 0, lax.bitwise_xor, (axis,))`` but built only from
    ``sum`` / shifts / bitwise-and, because the ``bitwise_xor`` reduction primitive fails to
    legalize on the Apple Metal (``jax-metal``) XLA backend
    (``UNIMPLEMENTED: failed to legalize operation 'mhlo.reduce'``). Numerically identical
    on every backend; used for Zobrist hashing so chess/go/etc. run on Apple Silicon GPUs.
    """
    nbits = jnp.iinfo(operand.dtype).bits
    bitpos = jnp.arange(nbits, dtype=operand.dtype)
    bits = (jnp.expand_dims(operand, -1) >> bitpos) & 1  # (..., nbits)
    parity = jnp.sum(bits, axis=axis) & 1  # reduce the requested axis, keep bit axis
    weights = jnp.ones((), operand.dtype) << bitpos
    return jnp.sum(parity.astype(operand.dtype) * weights, axis=-1).astype(operand.dtype)


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
