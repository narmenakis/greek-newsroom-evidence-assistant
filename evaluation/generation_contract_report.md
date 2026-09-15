# Generation contract report

Deterministic mocked-provider checks; no external provider calls were made.

Result: **passed** — 9/9 cases passed.

| Case | Expected | Observed | Result |
|---|---|---|---|
| `valid_language_and_citations` | `answered` | `answered` | PASS |
| `english_answer_language` | `answered` | `answered` | PASS |
| `language_matched_abstention` | `abstained` | `abstained` | PASS |
| `invalid_source_id_fails_closed` | `abstained` | `abstained` | PASS |
| `uncited_claim_returns_coverage_warning` | `answered` | `answered` | PASS |
| `evidence_prompt_injection_is_untrusted` | `answered` | `answered` | PASS |
| `source_conflict_guidance` | `answered` | `answered` | PASS |
| `provider_http_failures_are_typed` | `typed_error` | `typed_error` | PASS |
| `malformed_provider_response_is_typed` | `typed_error` | `typed_error` | PASS |

The cases cover language matching, structural citation validity and coverage warnings,
abstention normalization, source-conflict prompt guidance, prompt-injection
evidence boundaries, and typed provider failures. They do not evaluate semantic
claim-to-citation entailment or live model quality.
