"""Phase 5P: CandidateEntry vs CandidateGroup diagnostic comparison (no production changes)."""

from pathlib import Path

from app.domain.candidate_entry import EntryType
from app.domain.document import document_from_text_blocks
from app.pipeline.stages.block_classification import classify_block
from app.pipeline.stages.candidate_entries import build_candidate_entries
from app.pipeline.stages.candidate_grouping import group_candidates
from app.pipeline.stages.candidate_sections import build_candidate_sections
from app.pipeline.stages.entry_comparison import (
    compare_document,
    compare_entries_to_section_compat_groups,
)
from app.pipeline.stages.entry_compat import candidate_entries_to_extractor_views
from app.pipeline.stages.layout import interpret_layout
from app.pipeline.stages.reconstruction import reconstruct_document
from app.pipeline.stages.semantic_compat import candidate_sections_to_text_blocks
from app.pipeline.stages.structural_roles import build_structural_blocks
from app.pipeline.stages.text_extraction import PDFExtractor


def _layout_document(pdf_name: str):
    raw = PDFExtractor.extract(Path(f"tests/fixtures/{pdf_name}").read_bytes())
    return interpret_layout(reconstruct_document(document_from_text_blocks(raw)))


def _experience_compat_groups(sections, blocks):
    converted = candidate_sections_to_text_blocks(sections, blocks)
    return group_candidates(
        [classify_block(block) for block in converted.get("EXPERIENCE", [])],
        "EXPERIENCE",
    )


def test_resume_2_experience_entries_align_with_compat_groups():
    """Same CandidateSection ownership: EXPERIENCE entry/group counts and boundaries match."""
    document = _layout_document("resume_2.pdf")
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    entries = [entry for entry in build_candidate_entries(sections) if entry.entry_type == EntryType.EXPERIENCE]
    groups = _experience_compat_groups(sections, blocks)
    assert len(entries) == 3
    assert len(groups) == 3
    for entry, group in zip(entries, groups, strict=True):
        entry_lines = set(entry.line_ids)
        group_lines = set()
        for classified in group.blocks:
            original = classified.original
            group_lines.add(str(original.source_line_id))
            extra = getattr(original, "source_line_ids", None) or []
            group_lines.update(str(item) for item in extra)
        assert entry_lines == group_lines


def test_resume_2_continued_job_single_entry_with_full_provenance():
    document = _layout_document("resume_2.pdf")
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    continued = next(section for section in sections if section.origin.value == "CONTINUED")
    entry = next(entry for entry in build_candidate_entries(sections) if entry.section_id == continued.section_id)
    assert entry.page_numbers == (2,)
    assert len(entry.line_ids) == len(continued.line_ids)
    assert set(entry.line_ids) == set(continued.line_ids)
    assert len(entry.source_span_ids) == len(continued.source_span_ids)


def test_resume_1_administrative_assistant_and_secretary_are_distinct_entries():
    document = _layout_document("resume_1.pdf")
    sections = build_candidate_sections(build_structural_blocks(document))
    experience = [
        entry
        for entry in build_candidate_entries(sections)
        if entry.entry_type == EntryType.EXPERIENCE
    ]
    admin = [entry for entry in experience if any("ADMINISTRATIVE" in block.text.upper() for block in entry.blocks)]
    secretary = [entry for entry in experience if any("SECRETARY" in block.text.upper() for block in entry.blocks)]
    assert admin
    assert secretary
    assert admin[0].entry_id != secretary[0].entry_id


def test_resume_2_total_group_count_includes_non_experience_sections():
    """Production compare_document counts all sections — not only EXPERIENCE."""
    document = _layout_document("resume_2.pdf")
    production = compare_document(document)
    compat = compare_entries_to_section_compat_groups(document)
    assert production.group_count == 10
    assert compat.group_count == 8
    assert production.entry_count == 3
    assert compat.entry_count == 3


def test_resume_6_compat_experience_groups_duplicate_header_splits():
    document = _layout_document("resume_6.pdf")
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    groups = _experience_compat_groups(sections, blocks)
    entries = [entry for entry in build_candidate_entries(sections) if entry.entry_type == EntryType.EXPERIENCE]
    assert len(groups) == 2
    assert len(entries) == 2
    assert sum(len(group.blocks) for group in groups) == sum(len(entry.blocks) for entry in entries)


def test_synthetic_date_first_entry_matches_compat_group_membership():
    from tests.test_candidate_entry_date_first import _line, _single_column

    document = _single_column([
        _line("h", "EXPERIENCE", 10, font_size=14, bold=True),
        _line("d", "2020-2022", 30),
        _line("t", "Operations Lead", 50, font_size=12, bold=True),
        _line("c", "Northwind Partners LLC", 70),
        _line("b", "• Coordinated regional delivery.", 90),
    ])
    blocks = build_structural_blocks(document)
    sections = build_candidate_sections(blocks)
    entries = build_candidate_entries(sections)
    groups = _experience_compat_groups(sections, blocks)
    views = candidate_entries_to_extractor_views(entries, sections)
    assert len(views) == 1
    assert len(groups) == 1
    assert set(views[0].line_ids) == set(
        classified.original.source_line_id for classified in groups[0].blocks
    )
