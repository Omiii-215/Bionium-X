"""
BioniumXObject — Universal base class for all Bionium-X data structures.
Modeled after Stingray's StingrayObject pattern.
"""
from typing import List, Dict, Any, Tuple, Type, TypeVar, Optional, Union
import numpy as np
import warnings


T = TypeVar('T', bound='BioniumXObject')

class BioniumXObject:
    """
    Base class for all Bionium-X scientific data objects.

    Provides standard I/O, arithmetic, and metadata handling
    for TransmissionSpectrum, EmissionSpectrum, and derived products.

    Parameters
    ----------
    Not instantiated directly. Use subclasses.

    Notes
    -----
    All subclasses must define ``_required_attrs`` listing the array
    attributes that must be set for the object to be valid.
    """

    _required_attrs: List[str] = []      
    _metadata_attrs: List[str] = []       

    def __repr__(self) -> str:
        name: str = self.__class__.__name__
        attrs: Dict[str, Any] = {a: getattr(self, a, None) for a in self._required_attrs}
        shape_info: Dict[str, Union[Tuple[int, ...], Any]] = {
            k: (v.shape if hasattr(v, 'shape') else v)
            for k, v in attrs.items() if v is not None
        }
        return f"<{name} | {shape_info}>"

    def _validate(self) -> None:
        """Check all required attributes are set and consistent."""
        for attr in self._required_attrs:
            val: Optional[Any] = getattr(self, attr, None)
            if val is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires '{attr}' to be set."
                )

    def write(self, filename: str, fmt: str = "hdf5") -> None:
        """
        Save this object to a file.

        Parameters
        ----------
        filename : str
            Output file path.
        fmt : str, optional
            Output format. Only ``'hdf5'`` is currently supported (the
            default). Support for ``'fits'`` and ``'ascii'`` is planned but
            not yet implemented.

        Raises
        ------
        NotImplementedError
            If ``fmt`` is not one of the currently supported formats.

        Examples
        --------
        >>> spec.write("K2-18b.h5")
        >>> spec.write("K2-18b.h5", fmt="hdf5")
        """
        from bioniumx.io import write_object
        write_object(self, filename, fmt=fmt)

    @classmethod
    def read(cls: Type[T], filename: str, fmt: str = "hdf5") -> T:
        """
        Load an object from a file.

        Parameters
        ----------
        filename : str
            Input file path.
        fmt : str, optional
            Input format. Only ``'hdf5'`` is currently supported (the
            default). Support for ``'fits'`` and ``'ascii'`` is planned but
            not yet implemented.

        Returns
        -------
        obj : BioniumXObject subclass
            Reconstructed object.

        Raises
        ------
        NotImplementedError
            If ``fmt`` is not one of the currently supported formats.

        Examples
        --------
        >>> spec = TransmissionSpectrum.read("K2-18b.h5")
        """
        from bioniumx.io import read_object
        return read_object(cls, filename, fmt=fmt)

    def apply_wavelength_mask(self: T, wl_min: float, wl_max: float) -> T:
        """
        Return a masked copy of this object within [wl_min, wl_max] microns.

        Parameters
        ----------
        wl_min, wl_max : float
            Wavelength bounds in microns.

        Returns
        -------
        masked : T
            A masked copy retaining the subclass's specific type.
        """
        raise NotImplementedError("Subclasses must implement apply_wavelength_mask")