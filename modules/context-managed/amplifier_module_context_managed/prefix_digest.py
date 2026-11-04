"""Bounded, content-checked reuse of a canonical checkpoint prefix digest."""
import hashlib
import math
import struct
import sys


class _Uncached(Exception):
    pass


class _BudgetExceeded(_Uncached):
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
        self._prefix_hash = None

    def digest(self, value, canonical_digest, canonical_bytes=None):
        if self._prefix_hash is not None and type(value) is list and canonical_bytes is not None:
            return self._partial(value, canonical_digest, canonical_bytes)
        witness_builder = _Witness(self.LIMIT)
        try:
            # Integer conversion settings can change JSON validity between calls.
            witness = witness_builder.owned((witness_builder.owned(sys.get_int_max_str_digits()),
                                             witness_builder.freeze(value)))
            if self.sha256 is not None and witness == self.witness:
                self.witness, self.charge = witness, witness_builder.charge
                return self.sha256
        except _BudgetExceeded:
            if type(value) is list and canonical_bytes is not None:
                return self._partial(value, canonical_digest, canonical_bytes)
            self.clear()
            return canonical_digest(value)
        except (_Uncached, RecursionError):
            self.clear()
            return canonical_digest(value)
        self.clear()
        result = canonical_digest(value)
        self.witness, self.sha256, self.charge = witness, result, witness_builder.charge
        return result

    def _partial(self, value, canonical_digest, canonical_bytes):
        # Cache complete leading rows, never a fragment of a message. The tail is
        # still canonically serialized on every export. No history is changed.
        builder = _Witness(self.LIMIT)
        state = hashlib.sha256(b"[")
        rows = []
        try:
            builder.owned(state)
            limit = builder.owned(sys.get_int_max_str_digits())
            builder.owned((limit, ()))
            builder.owned(())
            builder.active.add(id(value))
            for row in value:
                before = builder.charge
                try:
                    builder.charge += struct.calcsize("P")  # one retained tuple slot
                    if builder.charge > self.LIMIT:
                        raise _BudgetExceeded()
                    rows.append(builder.freeze(row))
                except _BudgetExceeded:
                    builder.charge = before
                    break
            count = len(rows)
            if not count:
                raise _Uncached()
            # Inspect the rest without retaining another witness. This rejects
            # custom hooks/cycles before any piecemeal serialization can run.
            for index in range(count, len(value)):
                builder.freeze(value[index], retain=False)
            witness = (limit, tuple(rows))
            if self._prefix_hash is not None and witness == self.witness:
                state = self._prefix_hash.copy()
            else:
                for index in range(count):
                    if index:
                        state.update(b",")
                    state.update(canonical_bytes(value[index]))
            result_hash = state.copy()
            if count < len(value):
                tail = canonical_bytes(value[count:])
                result_hash.update(b",")
                result_hash.update(tail[1:])  # tail includes the final list bracket
            else:
                result_hash.update(b"]")
            result = result_hash.hexdigest()
        except (_Uncached, RecursionError, ValueError, TypeError, UnicodeError):
            self.clear()
            # Built-in invalid values keep the whole serializer's exact error
            # (including UTF-8 position/recursion errors). Unsupported hooks have
            # not run yet, and run only through the original whole-value path.
            return canonical_digest(value)
        self.witness, self.sha256, self.charge = witness, result, builder.charge
        self._prefix_hash = state
        return result


class _Witness:
    def __init__(self, limit):
        self.limit = limit
        self.charge = 0
        self.active = set()

    def owned(self, item):
        self.charge += sys.getsizeof(item)
        if self.charge > self.limit:
            raise _BudgetExceeded()
        return item

    def freeze(self, item, retain=True):
        kind = type(item)
        if kind is str or item is None:
            return item
        if kind in (int, bool):
            return self.owned((kind, item)) if retain else None
        if kind is float:
            if not math.isfinite(item):
                raise _Uncached()
            return self.owned((kind, self.owned(struct.pack("!d", item)))) if retain else None
        if kind not in (dict, list, tuple) or id(item) in self.active:
            raise _Uncached()
        if retain and len(item) > (self.limit - self.charge) // struct.calcsize("P"):
            raise _BudgetExceeded()
        self.active.add(id(item))
        try:
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise _Uncached()
                if retain:
                    children = self.owned(tuple(self.owned((key, self.freeze(item[key]))) for key in sorted(item)))
                else:
                    for child in item.values():
                        self.freeze(child, retain=False)
            elif retain:
                children = self.owned(tuple(self.freeze(child) for child in item))
            else:
                for child in item:
                    self.freeze(child, retain=False)
            return self.owned((kind, children)) if retain else None
        finally:
            self.active.remove(id(item))
