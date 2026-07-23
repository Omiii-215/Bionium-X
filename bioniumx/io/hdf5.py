"""
File input/output for BioniumX objects.

Only the HDF5 format is currently implemented. Support for ``'fits'`` and
``'ascii'`` is planned but not yet available; requesting an unsupported format
raises :class:`NotImplementedError` with a message listing the formats that
are currently supported.
"""
import h5py
import numpy as np

#: File formats currently implemented by :func:`write_object` / :func:`read_object`.
SUPPORTED_FORMATS = ("hdf5",)


def _unsupported_format_error(fmt: str) -> NotImplementedError:
    """Build a clear, actionable error for an I/O format that is not yet supported."""
    supported = ", ".join(repr(f) for f in SUPPORTED_FORMATS)
    return NotImplementedError(
        f"Format {fmt!r} is not supported yet. "
        f"Currently supported formats: {supported}. "
        "Support for 'fits' and 'ascii' is planned."
    )


def write_object(obj, filename: str, fmt: str = "hdf5"):
    """
    Write a BioniumXObject to a file.

    Only ``fmt='hdf5'`` is currently supported; any other value raises
    :class:`NotImplementedError`.
    """
    if fmt == "hdf5":
        with h5py.File(filename, "w") as f:
            f.attrs["class_name"] = obj.__class__.__name__
            for attr in obj._required_attrs:
                f.create_dataset(attr, data=getattr(obj, attr))
            if hasattr(obj, "err"):
                f.create_dataset("err", data=obj.err)

            meta_group = f.create_group("meta")
            for k, v in obj.meta.items():
                if v is not None:
                    meta_group.attrs[k] = v
    else:
        raise _unsupported_format_error(fmt)


def read_object(cls, filename: str, fmt: str = "hdf5"):
    """
    Read a BioniumXObject from a file.

    Only ``fmt='hdf5'`` is currently supported; any other value raises
    :class:`NotImplementedError`.
    """
    if fmt == "hdf5":
        with h5py.File(filename, "r") as f:
            kwargs = {}
            for attr in cls._required_attrs:
                kwargs[attr] = f[attr][:]
            if "err" in f:
                kwargs["err"] = f["err"][:]

            if "meta" in f:
                for k, v in f["meta"].attrs.items():
                    kwargs[k] = v

            return cls(**kwargs)
    else:
        raise _unsupported_format_error(fmt)
