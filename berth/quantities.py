"""Typed quantities for the values that have been passed wrongly.

Two of the eight instrument defects in this project were the same shape: a
number handed to a function that expected a different number of the same
shape. A microbenchmarked device-to-device copy rate was passed where a
datasheet peak was expected, and a dimensionless efficiency was fitted on one
card's timings while being divided by another card's bandwidth. Both produced
confident output. One reported a clean file as contaminated; the other
reported 109.5 percent error with a fitted efficiency of 1.761.

Neither was a careless mistake and neither would have been caught by review.
They were caught because the resulting numbers were absurd, which is luck
rather than process: at 0.62 instead of 1.761 the second would have been
written up as a finding.

A float carries no information about what it means. These types do. They
carry the device the value belongs to and where it came from, and the
operations that would have produced both defects raise instead of returning a
plausible number.

    >>> peak = Ceiling.datasheet("h100-pcie", bandwidth_tbs=2.0)
    >>> micro = Ceiling.microbench("h100-pcie", bandwidth_tbs=1.74)
    >>> peak.efficiency_of(measured_tbs=1.7)
    0.85
    >>> micro.efficiency_of(measured_tbs=1.7)
    Traceback (most recent call last):
    UnitError: efficiency must be taken against a datasheet peak
"""

from dataclasses import dataclass


class UnitError(ValueError):
    """A quantity was used where a different quantity was meant."""


@dataclass(frozen=True)
class Ceiling:
    """An upper bound on a device's rate, tagged with device and source.

    `source` matters because the two are not interchangeable and the
    difference is not visible in the number. A datasheet peak is what a vendor
    quotes. A microbenchmark is what a copy loop achieves, and for bandwidth
    it typically lands at 70 to 90 percent of the datasheet figure. Effective
    bandwidth is conventionally reported as a fraction of the datasheet peak,
    so computing it against a microbenchmark produces values above 1.0, which
    is what defect 5 did on every batch of a clean file.
    """

    device: str
    source: str                      # "datasheet" | "microbench"
    bandwidth_tbs: float | None = None
    compute_tflops: float | None = None

    SOURCES = ("datasheet", "microbench")

    def __post_init__(self):
        if self.source not in self.SOURCES:
            raise UnitError(f"source must be one of {self.SOURCES}, "
                            f"got {self.source!r}")
        if self.bandwidth_tbs is None and self.compute_tflops is None:
            raise UnitError("a ceiling with neither bandwidth nor compute "
                            "describes nothing")
        if self.bandwidth_tbs is not None and self.bandwidth_tbs > 100:
            raise UnitError(
                f"bandwidth_tbs={self.bandwidth_tbs} is implausible as TB/s. "
                f"The fastest accelerator shipping is under 10 TB/s. This is "
                f"almost certainly GB/s: divide by 1000.")

    @classmethod
    def datasheet(cls, device, **kw):
        return cls(device=device, source="datasheet", **kw)

    @classmethod
    def microbench(cls, device, **kw):
        return cls(device=device, source="microbench", **kw)

    def efficiency_of(self, measured_tbs: float, device: str | None = None):
        """Fraction of this ceiling that a measurement achieved.

        Refuses two things. A microbenchmark in the denominator, because the
        convention is a datasheet fraction and mixing them silently changes
        what the number means. And a measurement from a different device,
        because a dimensionless efficiency is only meaningful against the
        bandwidth of the card that produced the timing.
        """
        if self.source != "datasheet":
            raise UnitError(
                "efficiency must be taken against a datasheet peak, not a "
                f"{self.source}. A copy benchmark understates achievable read "
                "bandwidth, so dividing by it gives efficiencies above 1.0. "
                "Pass Ceiling.datasheet(...), or use headroom_of() if you "
                "genuinely want the fraction of measured ceiling.")
        if self.bandwidth_tbs is None:
            raise UnitError("this ceiling carries no bandwidth")
        if device is not None and device != self.device:
            raise UnitError(
                f"measurement is from {device!r} and this ceiling is for "
                f"{self.device!r}. A dimensionless efficiency has to be "
                f"fitted against the bandwidth of the card that produced the "
                f"timing. Fitting across a device boundary is what produced a "
                f"reported efficiency of 1.761.")
        return measured_tbs / self.bandwidth_tbs

    def headroom_of(self, measured_tbs: float, device: str | None = None):
        """Fraction of a microbenchmarked ceiling that was achieved.

        The honest name for the other operation. Values near 1.0 are expected
        and values slightly above it are not alarming, because a copy loop is
        not the fastest possible access pattern.
        """
        if self.source != "microbench":
            raise UnitError("headroom is measured against a microbenchmark; "
                            "use efficiency_of() for a datasheet peak")
        if device is not None and device != self.device:
            raise UnitError(f"measurement is from {device!r}, ceiling is for "
                            f"{self.device!r}")
        return measured_tbs / self.bandwidth_tbs


def transfer_efficiency(fitted_on: Ceiling, fitted_value: float,
                        applied_to: Ceiling) -> float:
    """Carry a dimensionless efficiency from one device to another.

    This is the only legitimate way to use one card's constant on another, and
    it is the operation a leave-one-silicon-out split performs. The efficiency
    is fitted against the training card's own peak and then multiplied by the
    held-out card's peak. Doing it in one step, as a raw float, is what defect
    6 did in two.
    """
    for c in (fitted_on, applied_to):
        if c.source != "datasheet":
            raise UnitError("transfer requires datasheet peaks on both sides")
        if c.bandwidth_tbs is None:
            raise UnitError(f"{c.device} ceiling carries no bandwidth")
    if not 0.0 < fitted_value <= 1.2:
        raise UnitError(
            f"efficiency of {fitted_value:.3f} is not a fraction of a peak. "
            f"Above 1.0 means the denominator was wrong, most often a "
            f"microbenchmark used as a datasheet figure or a fit taken across "
            f"two devices.")
    return fitted_value * applied_to.bandwidth_tbs
