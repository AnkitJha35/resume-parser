from app.extractors.date_parser import DateRangeParser


def test_date_parser_parses_month_year_range():
    result = DateRangeParser.parse("Jan 2022 - Present")

    assert result is not None
    assert result.startDate == "2022-01"
    assert result.endDate is None
    assert result.current is True


def test_date_parser_parses_full_month_names():
    result = DateRangeParser.parse("January 2022 - March 2024")

    assert result is not None
    assert result.startDate == "2022-01"
    assert result.endDate == "2024-03"
    assert result.current is False


def test_date_parser_parses_year_only_range():
    result = DateRangeParser.parse("2022 - 2024")

    assert result is not None
    assert result.startDate == "2022"
    assert result.endDate == "2024"
    assert result.current is False


def test_date_parser_parses_numeric_month_range():
    result = DateRangeParser.parse("06/2022 - Present")

    assert result is not None
    assert result.startDate == "2022-06"
    assert result.endDate is None
    assert result.current is True


def test_date_parser_parses_short_month_current_range():
    result = DateRangeParser.parse("Jun 2022 - Current")

    assert result is not None
    assert result.startDate == "2022-06"
    assert result.endDate is None
    assert result.current is True


def test_date_parser_returns_none_for_malformed_input():
    result = DateRangeParser.parse("invalid range")

    assert result is None


def test_date_parser_handles_labelled_date_colon_space():
    result = DateRangeParser.parse("Date : 07/2022 - Present")

    assert result is not None
    assert result.startDate == "2022-07"
    assert result.endDate is None
    assert result.current is True


def test_date_parser_handles_labelled_date_colon_no_space():
    result = DateRangeParser.parse("Date: 07/2022 - Present")

    assert result is not None
    assert result.startDate == "2022-07"
    assert result.endDate is None
    assert result.current is True


def test_date_parser_handles_labelled_date_lowercase():
    result = DateRangeParser.parse("date : Jan 2022 - Present")

    assert result is not None
    assert result.startDate == "2022-01"
    assert result.endDate is None
    assert result.current is True


def test_date_parser_ignores_zero_width_trailing_chars():
    result = DateRangeParser.parse("Date : 02/2022 - 06/2022 \u200b")

    assert result is not None
    assert result.startDate == "2022-02"
    assert result.endDate == "2022-06"
    assert result.current is False

    result_two = DateRangeParser.parse("Date : 07/2019 - 06/2022 \u200b")

    assert result_two is not None
    assert result_two.startDate == "2019-07"
    assert result_two.endDate == "2022-06"
    assert result_two.current is False
