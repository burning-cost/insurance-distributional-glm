"""Tests for link functions."""
import numpy as np
import pytest
from insurance_distributional_glm.families.base import (
    LogLink, LogitLink, IdentityLink, SqrtLink,
)

LINKS = [LogLink(), LogitLink(), IdentityLink(), SqrtLink()]


# ---------------------------------------------------------------------------
# LogLink
# ---------------------------------------------------------------------------

class TestLogLink:
    link = LogLink()

    def test_name(self):
        assert self.link.name == "log"

    def test_link_positive(self):
        theta = np.array([1.0, 2.0, 10.0])
        np.testing.assert_allclose(self.link.link(theta), np.log(theta))

    def test_inverse_roundtrip(self):
        theta = np.array([0.5, 1.0, 5.0, 100.0])
        eta = self.link.link(theta)
        np.testing.assert_allclose(self.link.inverse(eta), theta, rtol=1e-10)

    def test_deriv_matches_numerical(self):
        theta = np.array([1.0, 2.0, 5.0])
        h = 1e-6
        numerical = (self.link.link(theta + h) - self.link.link(theta - h)) / (2 * h)
        np.testing.assert_allclose(self.link.deriv(theta), numerical, rtol=1e-5)

    def test_inverse_deriv_matches_numerical(self):
        eta = np.array([0.0, 1.0, -1.0, 2.0])
        h = 1e-6
        numerical = (self.link.inverse(eta + h) - self.link.inverse(eta - h)) / (2 * h)
        np.testing.assert_allclose(self.link.inverse_deriv(eta), numerical, rtol=1e-5)

    def test_clipping_extreme_eta(self):
        # Should not raise
        eta_large = np.array([100.0, -100.0])
        result = self.link.inverse(eta_large)
        assert np.all(np.isfinite(result))


class TestLogitLink:
    link = LogitLink()

    def test_name(self):
        assert self.link.name == "logit"

    def test_link_probability(self):
        theta = np.array([0.1, 0.5, 0.9])
        expected = np.log(theta / (1 - theta))
        np.testing.assert_allclose(self.link.link(theta), expected, rtol=1e-10)

    def test_inverse_is_sigmoid(self):
        eta = np.array([0.0, 1.0, -1.0])
        expected = 1.0 / (1.0 + np.exp(-eta))
        np.testing.assert_allclose(self.link.inverse(eta), expected, rtol=1e-10)

    def test_inverse_bounds(self):
        theta = self.link.inverse(np.array([0.0, 1.0, -1.0]))
        assert np.all(theta > 0) and np.all(theta < 1)

    def test_roundtrip(self):
        theta = np.array([0.1, 0.3, 0.7, 0.9])
        np.testing.assert_allclose(
            self.link.inverse(self.link.link(theta)), theta, rtol=1e-8
        )

    def test_deriv_matches_numerical(self):
        theta = np.array([0.2, 0.5, 0.8])
        h = 1e-6
        numerical = (self.link.link(theta + h) - self.link.link(theta - h)) / (2 * h)
        np.testing.assert_allclose(self.link.deriv(theta), numerical, rtol=1e-4)

    def test_inverse_deriv_matches_numerical(self):
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        h = 1e-6
        numerical = (self.link.inverse(eta + h) - self.link.inverse(eta - h)) / (2 * h)
        np.testing.assert_allclose(self.link.inverse_deriv(eta), numerical, rtol=1e-5)


class TestIdentityLink:
    link = IdentityLink()

    def test_name(self):
        assert self.link.name == "identity"

    def test_link_is_identity(self):
        x = np.array([1.0, -2.0, 3.5])
        np.testing.assert_array_equal(self.link.link(x), x)

    def test_inverse_is_identity(self):
        x = np.array([1.0, -2.0, 3.5])
        np.testing.assert_array_equal(self.link.inverse(x), x)

    def test_deriv_is_ones(self):
        x = np.array([1.0, 5.0, 100.0])
        np.testing.assert_array_equal(self.link.deriv(x), np.ones_like(x))

    def test_inverse_deriv_is_ones(self):
        eta = np.array([0.0, 1.0, -5.0])
        np.testing.assert_array_equal(self.link.inverse_deriv(eta), np.ones_like(eta))


class TestSqrtLink:
    link = SqrtLink()

    def test_name(self):
        assert self.link.name == "sqrt"

    def test_roundtrip(self):
        theta = np.array([1.0, 4.0, 9.0, 16.0])
        np.testing.assert_allclose(
            self.link.inverse(self.link.link(theta)), theta, rtol=1e-10
        )

    def test_inverse_non_negative(self):
        eta = np.array([0.5, 1.0, 2.0])
        assert np.all(self.link.inverse(eta) >= 0)
