from tests.conftest import require_fixture
from pathlib import Path

from app.pipeline.stages.text_extraction import PDFExtractor
from app.pipeline.stages.reading_order import ReadingOrder


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> bytes:
    return require_fixture(name).read_bytes()



def test_reading_order_single_column_preserves_order():
    data = load_fixture("single-column.pdf")
    blocks = PDFExtractor.extract(data)

    reordered = ReadingOrder.reorder(blocks)

    # Single-column pages should preserve top-to-bottom order per page
    original_sorted = sorted(blocks, key=lambda b: (b.page_number, b.y0, b.x0))
    assert [b.text for b in reordered] == [b.text for b in original_sorted]


def test_reading_order_multi_column_swe_experienced():
    data = load_fixture("swe_experienced_resume.pdf")
    blocks = PDFExtractor.extract(data)

    # Ensure clustering detects multiple columns on at least one page
    # and that reordered output places left-column blocks before right-column ones
    reordered = ReadingOrder.reorder(blocks)

    # Group original blocks by page
    page1_original = [b for b in blocks if b.page_number == 1]
    clusters = ReadingOrder.cluster_blocks_by_x0(page1_original)
    assert len(clusters) >= 2

    # Identify cluster membership by object identity
    cluster0_ids = {id(b) for b in clusters[0]}
    cluster1_ids = {id(b) for b in clusters[1]}

    # In the reordered list for page 1, all cluster0 blocks should appear before cluster1 blocks
    page1_reordered = [b for b in reordered if b.page_number == 1]
    first_index_cluster1 = next((i for i, b in enumerate(page1_reordered) if id(b) in cluster1_ids), None)
    last_index_cluster0 = max((i for i, b in enumerate(page1_reordered) if id(b) in cluster0_ids), default=-1)

    assert first_index_cluster1 is not None
    assert last_index_cluster0 < first_index_cluster1
