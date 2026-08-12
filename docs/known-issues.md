## ContactExtractor: location field
- ContactExtractor sometimes returns a fragment of the summary/profile paragraph as `location` when no real location exists on the resume.
- Expected: should return `null` when no confident location signal is found, rather than falling back to nearby text.
- Repro: tests/fixtures/fresher_hr_resume.pdf
- Priority: low — doesn't corrupt other fields or break parsing.
