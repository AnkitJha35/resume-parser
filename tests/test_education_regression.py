from app.extractors.education import EducationExtractor


def test_extract_compact_degree_and_institution():
    extractor = EducationExtractor()
    lines = [
        "M.C.A NIT Calicut",
        "Date: 07/2019 - 06/2022",
        "Location: Kerala, India",
        "",
        "B.SC-IT Magadh University",
        "Date: 07/2015 - 08/2018",
        "Location: Patna, India",
    ]

    # Use group-based extraction to mirror parser behavior (classification labels)
    from app.pipeline.stages.block_classification import ClassifiedBlock
    from app.pipeline.stages.candidate_grouping import CandidateGroup
    from app.pipeline.stages.text_extraction import TextBlock

    cb1 = ClassifiedBlock(original=TextBlock(text="M.C.A NIT Calicut", page_number=1, x0=0, y0=0, x1=0, y1=0), label="UNKNOWN", score=0.0, reasons=[])
    cb2 = ClassifiedBlock(original=TextBlock(text="Date: 07/2019 - 06/2022", page_number=1, x0=0, y0=1, x1=0, y1=0), label="DATE", score=1.0, reasons=[])
    cb3 = ClassifiedBlock(original=TextBlock(text="Location: Kerala, India", page_number=1, x0=0, y0=2, x1=0, y1=0), label="LOCATION", score=0.5, reasons=[])

    group1 = CandidateGroup(section="EDUCATION", blocks=[cb1, cb2, cb3], page_number=1, column_id=0, start_index=0, end_index=2, summary_text="M.C.A NIT Calicut")

    cb4 = ClassifiedBlock(original=TextBlock(text="B.SC-IT Magadh University", page_number=1, x0=0, y0=3, x1=0, y1=0), label="UNKNOWN", score=0.0, reasons=[])
    cb5 = ClassifiedBlock(original=TextBlock(text="Date: 07/2015 - 08/2018", page_number=1, x0=0, y0=4, x1=0, y1=0), label="DATE", score=1.0, reasons=[])
    cb6 = ClassifiedBlock(original=TextBlock(text="Location: Patna, India", page_number=1, x0=0, y0=5, x1=0, y1=0), label="LOCATION", score=0.5, reasons=[])

    group2 = CandidateGroup(section="EDUCATION", blocks=[cb4, cb5, cb6], page_number=1, column_id=0, start_index=3, end_index=5, summary_text="B.SC-IT Magadh University")

    entries = extractor.extract(groups=[group1, group2])

    assert len(entries) >= 2
    # first entry
    e1 = entries[0]
    assert e1["degree"] == "M.C.A"
    assert e1["institution"] == "NIT Calicut"
    assert e1["startDate"] == "2019-07"
    assert e1["endDate"] == "2022-06"

    # second entry
    e2 = entries[1]
    assert e2["degree"] == "B.SC-IT"
    assert e2["institution"] == "Magadh University"
    assert e2["startDate"] == "2015-07"
    assert e2["endDate"] == "2018-08"
