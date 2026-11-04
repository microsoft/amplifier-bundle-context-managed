"""Bounded, content-checked reuse of a canonical checkpoint prefix digest."""
import math
import struct
import sys


class _Uncached(Exception):
    pass


class PrefixDigest:
    # This bounds private tuple/float witness storage, not history or requests.
    # Immutable source strings/keys are referenced, never copied. The context
    # clears this owner before replacing history; each hit replaces its witness
    # too, so equal freshly allocated strings do not retain superseded objects.
    LIMIT = 8 * 1024 * 1024

    def __init__(self):
        self.clear()

    def clear(self):
        self.witness = None
        self.sha256 = None
        self.charge = 0

    def digest(self, value, canonical_digest):
        charge = 0
        active = set()

        def owned(item):
            nonlocal charge
            charge += sys.getsizeof(item)
            if charge > self.LIMIT:
                raise _Uncached()
            return item

        def freeze(item):
            kind = type(item)
            if kind is str or item is None:
                return item
            if kind in (int, bool):
                return owned((kind, item))
            if kind is float:
                if not math.isfinite(item):
                    raise _Uncached()
                # Preserve JSON distinctions: bool/int/float and signed zero.
                return owned((kind, owned(struct.pack("!d", item))))
            if kind not in (dict, list, tuple) or id(item) in active:
                raise _Uncached()
            # Avoid allocating a huge flat witness before discovering no room.
            if len(item) > (self.LIMIT - charge) // struct.calcsize("P"):
                raise _Uncached()
            active.add(id(item))
            try:
                if kind is dict:
                    if any(type(key) is not str for key in item):
                        raise _Uncached()
                    children = owned(tuple(owned((key, freeze(item[key]))) for key in sorted(item)))
                else:
                    children = owned(tuple(freeze(child) for child in item))
                return owned((kind, children))
            finally:
                active.remove(id(item))

        try:
            # Python's process-wide integer conversion limit can make the same
            # exact integer fail JSON encoding after an earlier successful hash.
            witness = owned((owned(sys.get_int_max_str_digits()), freeze(value)))
            if self.sha256 is not None and witness == self.witness:
                self.witness, self.charge = witness, charge
                return self.sha256
        except (_Uncached, RecursionError):
            self.clear()
            # Custom containers, non-string keys, nonfinite floats and cycles
            # keep the original serializer's values, errors and side effects.
            return canonical_digest(value)
        self.clear()
        result = canonical_digest(value)
        self.witness, self.sha256, self.charge = witness, result, charge
        return result
