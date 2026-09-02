"""Provider-independent semantic extraction interface and mock implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.semantic_contract import (
    BlockClassification,
    DocumentArchetype,
    GroundedExperienceItem,
    GroundedPersonal,
    GroundedString,
    SemanticBlockCategory,
    SemanticInput,
    SemanticOutput,
)


class SemanticExtractionError(Exception):
    """Base exception for semantic extraction errors."""


class SemanticValidationError(SemanticExtractionError):
    """Raised when semantic output fails deterministic validation invariants."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            f"Semantic extraction validation failed with {len(violations)} violation(s): {'; '.join(violations)}"
        )


@runtime_checkable
class SemanticExtractor(Protocol):
    """Provider-independent interface for extracting semantic resume structures."""

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        """Extract structured resume entities from layout-aware SemanticInput."""
        ...


class MockSemanticExtractor:
    """Deterministic, provider-independent test double for the semantic extraction seam.

    Makes NO network calls, NO LLM calls, and contains NO heuristic parsing logic.
    Maps only explicit structural evidence (e.g. suggested_role) to demonstrative grounded
    entities to exercise provenance tracking, invariant validation, and Resume projection.
    """

    def extract(self, input_data: SemanticInput) -> SemanticOutput:
        name_item: GroundedString | None = None
        email_item: GroundedString | None = None
        phone_item: GroundedString | None = None
        block_classifications: list[BlockClassification] = []

        # 1. Personal Identity: Select from header blocks with explicit structural roles
        for b in input_data.blocks:
            text = b.text.strip()
            if not text:
                continue

            if b.page == 1 and b.region_kind == "header":
                if b.suggested_role == "HEADER" and name_item is None:
                    name_item = GroundedString(value=text, raw_value=text, source_block_ids=[b.block_id])
                    block_classifications.append(
                        BlockClassification(block_id=b.block_id, category=SemanticBlockCategory.PERSONAL)
                    )
                elif b.suggested_role == "CONTACT":
                    if "@" in text and email_item is None:
                        email_item = GroundedString(value=text, raw_value=text, source_block_ids=[b.block_id])
                        block_classifications.append(
                            BlockClassification(block_id=b.block_id, category=SemanticBlockCategory.PERSONAL)
                        )
                    elif any(ch.isdigit() for ch in text) and phone_item is None:
                        phone_item = GroundedString(value=text, raw_value=text, source_block_ids=[b.block_id])
                        block_classifications.append(
                            BlockClassification(block_id=b.block_id, category=SemanticBlockCategory.PERSONAL)
                        )

        personal = GroundedPersonal(
            name=name_item,
            email=email_item,
            phone=phone_item,
        )

        # 2. Demonstrative Experience: Map explicit ORGANIZATION, ENTRY_TITLE, or DATE roles
        exp_source_ids: list[str] = []
        company_item: GroundedString | None = None
        desig_item: GroundedString | None = None
        date_item: GroundedString | None = None

        org_block = next((b for b in input_data.blocks if b.region_kind != "header" and b.suggested_role == "ORGANIZATION"), None)
        if org_block:
            company_item = GroundedString(value=org_block.text, raw_value=org_block.text, source_block_ids=[org_block.block_id])
            exp_source_ids.append(org_block.block_id)
            block_classifications.append(
                BlockClassification(block_id=org_block.block_id, category=SemanticBlockCategory.EXPERIENCE)
            )

        title_block = next((b for b in input_data.blocks if b.region_kind != "header" and b.suggested_role == "ENTRY_TITLE"), None)
        if title_block:
            desig_item = GroundedString(value=title_block.text, raw_value=title_block.text, source_block_ids=[title_block.block_id])
            exp_source_ids.append(title_block.block_id)
            block_classifications.append(
                BlockClassification(block_id=title_block.block_id, category=SemanticBlockCategory.EXPERIENCE)
            )

        date_block = next((b for b in input_data.blocks if b.region_kind != "header" and b.suggested_role == "DATE"), None)
        if date_block:
            date_item = GroundedString(value=date_block.text, raw_value=date_block.text, source_block_ids=[date_block.block_id])
            exp_source_ids.append(date_block.block_id)
            block_classifications.append(
                BlockClassification(block_id=date_block.block_id, category=SemanticBlockCategory.EXPERIENCE)
            )

        experience: list[GroundedExperienceItem] = []
        if company_item or desig_item:
            experience.append(
                GroundedExperienceItem(
                    company=company_item,
                    designation=desig_item,
                    startDate=date_item,
                    source_block_ids=exp_source_ids,
                )
            )

        return SemanticOutput(
            document_archetype=DocumentArchetype.STANDARD_CV,
            block_classifications=block_classifications,
            personal=personal,
            experience=experience,
        )
