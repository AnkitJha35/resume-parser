"""Tests for structured form name order validation.

Verifies that human names extracted from structured forms (e.g., 'Last name: PARASHAR First name: JOSH')
can be canonically ordered ('JOSH PARASHAR') while strictly preventing:
- Uninverted/arbitrary permutations without form descriptors
- Hallucinated middle names or tokens
- Omitted/truncated name tokens
"""

import pytest
from app.domain.semantic_contract import (
    _is_name_form_descriptor_supported,
    _is_name_semantically_supported,
)


def test_structured_form_name_order_with_descriptors():
    """Form descriptor tokens (First name, Last name, Surname) permit token reordering."""
    # Josh Parashar style
    source = "Last name: PARASHAR First name: JOSH"
    assert _is_name_semantically_supported("JOSH PARASHAR", source)
    assert _is_name_semantically_supported("Josh Parashar", source)

    # Akibul Alam style
    source_alam = "Surname Alam First Name Akibul"
    assert _is_name_semantically_supported("Akibul Alam", source_alam)

    # Full name label prefix
    source_full = "Full Name: Jane Doe"
    assert _is_name_semantically_supported("Jane Doe", source_full)


def test_non_form_inverted_name_without_descriptors_fails():
    """Inverted name tokens without form descriptors are NOT accepted by descriptor support."""
    # "Smith John" without any descriptor like "Last name:" or "Surname:"
    source = "Smith John"
    # Form descriptor matching requires descriptor tokens
    assert not _is_name_form_descriptor_supported("John Smith", source)
    # And semantic support fails because exact substring also fails
    assert not _is_name_semantically_supported("John Smith", source)


def test_hallucinated_middle_name_fails():
    """Zero tokens may be added (no hallucinated middle names/words)."""
    source = "Last name: PARASHAR First name: JOSH"
    # Canonical has an extra token "Kumar" not present in source
    assert not _is_name_semantically_supported("JOSH KUMAR PARASHAR", source)
    assert not _is_name_form_descriptor_supported("JOSH KUMAR PARASHAR", source)


def test_omitted_token_fails():
    """Zero non-descriptor tokens may be omitted."""
    source = "Last name: PARASHAR First name: JOSH ALBERT"
    # Canonical omitted "Albert"
    assert not _is_name_semantically_supported("JOSH PARASHAR", source)
    assert not _is_name_form_descriptor_supported("JOSH PARASHAR", source)


def test_exact_substring_match_still_passes_without_descriptors():
    """Normal names matching exact substring pass through generic value support."""
    source = "John Doe is a software engineer"
    assert _is_name_semantically_supported("John Doe", source)


def test_multiblock_name_exact_permutation_passes():
    """When human name components come from multiple blocks (e.g. separate Surname and First Name cells),
    exact multiset permutation without form descriptor tokens is permitted."""
    source = "Alam Akibul"
    # Single block without descriptors fails:
    assert not _is_name_semantically_supported("Akibul Alam", source, is_multiblock=False)
    # Multi-block source with exact multiset equality passes:
    assert _is_name_semantically_supported("Akibul Alam", source, is_multiblock=True)
    # But hallucinated tokens in multi-block still fail:
    assert not _is_name_semantically_supported("Akibul Kumar Alam", source, is_multiblock=True)
    # And omitted tokens in multi-block still fail under form descriptor matching:
    assert not _is_name_form_descriptor_supported("Akibul", source, is_multiblock=True)

