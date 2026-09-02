"""Metadata and archetype classification for benchmark resume fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DocumentArchetype(str, Enum):
    STANDARD_CV = "standard_cv"
    MARITIME_CV = "maritime_cv"
    MARITIME_TABULAR = "maritime_tabular"
    STRUCTURED_FORM = "structured_form"


@dataclass(frozen=True)
class ResumeMetadata:
    filename: str
    archetype: DocumentArchetype
    candidate_name: str
    target_domain: str
    page_count_estimate: int
    has_tables: bool
    notes: str


BENCHMARK_FIXTURES: dict[str, ResumeMetadata] = {
    "AditCV_SOL.pdf": ResumeMetadata(
        filename="AditCV_SOL.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        candidate_name="Aditi Anand",
        target_domain="Human Resources",
        page_count_estimate=1,
        has_tables=False,
        notes="Standard 1-page HR Intern CV (Golden Resume A).",
    ),
    "fresher_hr_resume.pdf": ResumeMetadata(
        filename="fresher_hr_resume.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        candidate_name="Aditi Anand",
        target_domain="Human Resources",
        page_count_estimate=1,
        has_tables=False,
        notes="Standard 1-page Fresher HR CV (Golden Resume B).",
    ),
    "swe_experienced_resume.pdf": ResumeMetadata(
        filename="swe_experienced_resume.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        candidate_name="Ankit Jha",
        target_domain="Software Engineering",
        page_count_estimate=2,
        has_tables=False,
        notes="Standard 2-page Senior Software Engineer CV (Golden Resume C).",
    ),
    "Résume_Shubham.pdf": ResumeMetadata(
        filename="Résume_Shubham.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        candidate_name="Shubham Bajaj",
        target_domain="Software Engineering",
        page_count_estimate=2,
        has_tables=False,
        notes="Standard 2-page Software Engineer CV with embedded skills columns (Phase 7B).",
    ),
    "Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf": ResumeMetadata(
        filename="Rajeev_Ranjan_Prajapati_FullStack_Engineer.pdf",
        archetype=DocumentArchetype.STANDARD_CV,
        candidate_name="Rajeev Ranjan Prajapati",
        target_domain="Software Engineering",
        page_count_estimate=2,
        has_tables=False,
        notes="Standard 2-page Full-Stack Engineer CV with multi-page projects continuation.",
    ),
    "Sendrick Costa CV.pdf": ResumeMetadata(
        filename="Sendrick Costa CV.pdf",
        archetype=DocumentArchetype.MARITIME_CV,
        candidate_name="Sendrick Valentino Costa",
        target_domain="Maritime / Marine Engineering",
        page_count_estimate=3,
        has_tables=True,
        notes="Multi-page Marine Engineer CV containing sea-service tables, referee contacts, and certificates.",
    ),
    "2nd Officer Mayur Agarwal_062029.pdf": ResumeMetadata(
        filename="2nd Officer Mayur Agarwal_062029.pdf",
        archetype=DocumentArchetype.MARITIME_TABULAR,
        candidate_name="Mayur Agarwal",
        target_domain="Maritime / Deck Officer",
        page_count_estimate=4,
        has_tables=True,
        notes="Maritime 2nd Officer CV with extensive sea-service, STCW, and certification tables.",
    ),
    "AASHISH DG.pdf": ResumeMetadata(
        filename="AASHISH DG.pdf",
        archetype=DocumentArchetype.STRUCTURED_FORM,
        candidate_name="Aashish",
        target_domain="Maritime / Seafarer Registry",
        page_count_estimate=6,
        has_tables=True,
        notes="Official DG Shipping Seafarer Profile tabular verification report.",
    ),
    "AKIBUL ALAM CV(JO).pdf": ResumeMetadata(
        filename="AKIBUL ALAM CV(JO).pdf",
        archetype=DocumentArchetype.STRUCTURED_FORM,
        candidate_name="Akibul Alam",
        target_domain="Maritime / Deck Cadet",
        page_count_estimate=2,
        has_tables=True,
        notes="Pre-sea Nautical Science application bio-data form with next-of-kin and sea-service grids.",
    ),
    "CV Rishabh Dixit.pdf": ResumeMetadata(
        filename="CV Rishabh Dixit.pdf",
        archetype=DocumentArchetype.MARITIME_CV,
        candidate_name="Rishabh Dixit",
        target_domain="Maritime / Third Officer",
        page_count_estimate=2,
        has_tables=True,
        notes="Maritime CV with tabular sea-service records, STCW certifications, and declaration.",
    ),
    "JOSH PARASHAR MASTER CV2.pdf": ResumeMetadata(
        filename="JOSH PARASHAR MASTER CV2.pdf",
        archetype=DocumentArchetype.STRUCTURED_FORM,
        candidate_name="Josh Parashar",
        target_domain="Maritime / Master Mariner",
        page_count_estimate=4,
        has_tables=True,
        notes="Multi-page crew application form with vessel sea services, vaccinations, and passport grids.",
    ),
    "MUKUND 3RD OFF CV 2026.pdf": ResumeMetadata(
        filename="MUKUND 3RD OFF CV 2026.pdf",
        archetype=DocumentArchetype.MARITIME_CV,
        candidate_name="Mukund Kumar",
        target_domain="Maritime / Third Officer",
        page_count_estimate=2,
        has_tables=True,
        notes="3rd Officer maritime CV with structured sea-service experience table and STCW qualifications.",
    ),
}
