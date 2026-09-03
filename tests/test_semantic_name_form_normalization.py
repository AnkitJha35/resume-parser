"""Phase 8-7: Tests for deterministic form-label name normalization in semantic validation."""

from __future__ import annotations

import pytest

from app.domain.semantic_contract import (
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedString,
    NAME_FORM_DESCRIPTOR_TOKENS,
    SemanticBlockInput,
    SemanticInput,
    SemanticOutput,
    _is_name_form_descriptor_supported,
    _is_name_semantically_supported,
    _is_value_semantically_supported,
    validate_semantic_output,
)


def _make_semantic_input(blocks: list[tuple[str, str]]) -> SemanticInput:
    return SemanticInput(
        document_id="test_doc",
        page_count=1,
        pages=[],
        blocks=[
            SemanticBlockInput(
                block_id=bid,
                text=txt,
                page=1,
                bbox=[0.0, 0.0, 100.0, 20.0],
                region_id="r1",
                region_kind="header",
                reading_order=i,
            )
            for i, (bid, txt) in enumerate(blocks, 1)
        ],
    )


# =====================================================================
# 1. Helper Unit Tests for _is_name_form_descriptor_supported
# =====================================================================


def test_name_supported_with_separate_first_and_surname_blocks():
    """Akibul case: 'First Name Akibul Surname Alam' -> 'Akibul Alam'."""
    assert _is_name_form_descriptor_supported("Akibul Alam", "First Name Akibul Surname Alam") is True
    assert _is_name_semantically_supported("Akibul Alam", "First Name Akibul Surname Alam") is True


def test_name_supported_with_reversed_surname_first_blocks():
    """Josh case: 'Last name PARASHAR First name JOSH' -> 'JOSH PARASHAR'."""
    assert _is_name_form_descriptor_supported("JOSH PARASHAR", "Last name PARASHAR First name JOSH") is True
    assert _is_name_semantically_supported("JOSH PARASHAR", "Last name PARASHAR First name JOSH") is True


def test_name_supported_with_given_and_family_name_labels():
    """Given name / family name descriptors."""
    assert _is_name_form_descriptor_supported("Johnathan Doe", "Family Name: Doe, Given Name: Johnathan") is True
    assert _is_name_semantically_supported("Johnathan Doe", "Family Name: Doe, Given Name: Johnathan") is True


def test_name_supported_with_middle_name():
    """Middle name descriptor with multi-token names."""
    source = "First Name John Middle Name Robert Last Name Doe"
    assert _is_name_form_descriptor_supported("John Robert Doe", source) is True
    assert _is_name_semantically_supported("John Robert Doe", source) is True


def test_name_rejected_when_extra_hallucinated_middle_name():
    """Extra ungrounded tokens must fail exact multiset matching."""
    source = "Last name PARASHAR First name JOSH"
    # Canonical introduces ungrounded 'Kumar'
    assert _is_name_form_descriptor_supported("Josh Kumar Parashar", source) is False
    assert _is_name_semantically_supported("Josh Kumar Parashar", source) is False


def test_name_rejected_when_missing_name_token():
    """Missing tokens from grounded source blocks must fail multiset matching."""
    source = "Last name PARASHAR First name JOSH"
    # Canonical misses 'PARASHAR'
    assert _is_name_form_descriptor_supported("JOSH", source) is False


def test_name_rejected_when_completely_different_name():
    """Completely hallucinated name must fail."""
    source = "Last name PARASHAR First name JOSH"
    assert _is_name_form_descriptor_supported("John Doe", source) is False
    assert _is_name_semantically_supported("John Doe", source) is False


def test_standard_name_passes_direct_substring_first():
    """Normal single-word and multi-word names pass via direct substring check."""
    assert _is_name_semantically_supported("ADITI ANAND", "ADITI ANAND") is True
    assert _is_name_semantically_supported("Aditi Anand", "ADITI ANAND") is True
    assert _is_name_semantically_supported("Mayur", "Mayur Agarwal") is True


def test_case_and_punctuation_normalization():
    """Case, punctuation, and whitespace variations are supported."""
    source = "First-Name: [Aashish], Surname: [Gupta]"
    assert _is_name_semantically_supported("AASHISH GUPTA", source) is True
    assert _is_name_semantically_supported("aashish gupta", source) is True


def test_empty_or_descriptor_only_source_fails():
    """Source containing only descriptor words with no actual name tokens must return False."""
    assert _is_name_form_descriptor_supported("John", "First Name Last Name") is False
    assert _is_name_form_descriptor_supported("", "First Name Last Name") is False
    assert _is_name_form_descriptor_supported("First Name", "First Name") is False


# =====================================================================
# 2. Scope Restriction Tests: Non-Name Fields Must NOT Use Name Rules
# =====================================================================


def test_company_and_designation_do_not_use_name_form_normalization():
    """Company and designation fields strictly use _is_value_semantically_supported and forbid token permutation."""
    # Reordered company name must FAIL
    assert _is_value_semantically_supported("Industries ECE", "ECE Industries") is False
    # Descriptor word in company does not get stripped
    assert _is_value_semantically_supported("Google", "Company Name Google") is True  # exact substring
    assert _is_value_semantically_supported("Google First", "First Google") is False  # reordering rejected for company


def test_validate_semantic_output_enforces_name_rule_strictly_on_personal_name():
    """validate_semantic_output applies name normalization ONLY to personal.name."""
    sem_in = _make_semantic_input([
        ("b_name_1", "Surname Alam"),
        ("b_name_2", "First Name Akibul"),
        ("b_exp_1", "Company Industries ECE"),
    ])

    # Case A: personal.name reordered with descriptors -> VALID
    valid_output = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="Akibul Alam", source_block_ids=["b_name_1", "b_name_2"])
        )
    )
    violations = validate_semantic_output(valid_output, sem_in)
    assert violations == []

    # Case B: experience.company reordered with descriptors -> INVALID
    invalid_exp_output = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="Akibul Alam", source_block_ids=["b_name_1", "b_name_2"])
        ),
        experience=[
            GroundedExperienceItem(
                company=GroundedString(value="ECE Industries", source_block_ids=["b_exp_1"])  # source is 'Company Industries ECE'
            )
        ],
    )
    violations_exp = validate_semantic_output(invalid_exp_output, sem_in)
    assert any("UNSUPPORTED_CANONICAL_VALUE in experience[0].company" in v for v in violations_exp)


def test_document_title_rejection_still_fires_for_form_names():
    """Document titles like 'APPLICATION FORM' are rejected even if exact source blocks are referenced."""
    sem_in = _make_semantic_input([
        ("b_1", "APPLICATION FORM"),
    ])
    output = SemanticOutput(
        personal=GroundedPersonal(
            name=GroundedString(value="APPLICATION FORM", source_block_ids=["b_1"])
        )
    )
    violations = validate_semantic_output(output, sem_in)
    assert any("DOCUMENT_TITLE_AS_NAME" in v for v in violations)
