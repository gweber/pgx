import os
import sys
from urllib.request import urlopen

import jax
import jax.numpy as jnp
from jax import Array, lax

def _use_native_xor() -> bool:
    """True everywhere the native ``lax.bitwise_xor`` reduction legalizes (CPU/CUDA/ROCm/TPU);
    False only on Apple Metal (``jax-metal``), where it does not. Metal is detected by SIGNAL
    (a 'metal' platform or an 'apple' device_kind) rather than by guessing its exact backend
    string — a CUDA device reports platform 'gpu' with an NVIDIA device_kind, so it is never
    misread as Metal, and a Metal device that happened to report platform 'gpu' would still be
    caught by its Apple device_kind. Override with ``PGX_XOR_REDUCE=native|parity``."""
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

    Uses the native ``lax.reduce(operand, 0, lax.bitwise_xor, (axis,))`` on CPU/CUDA/TPU (a
    single reduction), and a sum/shift bit-parity fallback ONLY on Apple Metal, where the
    ``bitwise_xor`` reduction primitive fails to legalize
    (``UNIMPLEMENTED: failed to legalize operation 'mhlo.reduce'``). The fallback materializes
    a ``nbits``-wide bit expansion and is ~30-70x slower, so it must not be the default on
    CUDA/CPU. The branch is resolved at trace time (static, no runtime cost). Numerically
    identical on every backend; used for Zobrist hashing so chess/go/etc. also run on Apple
    Silicon GPUs.
    """
    if _use_native_xor():
        return lax.reduce(operand, jnp.zeros((), operand.dtype), lax.bitwise_xor, (axis,))
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
