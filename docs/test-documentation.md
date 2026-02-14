# Test Documentation

Generated from test modules with `python scripts/generate_test_docs.py`.

- Test files documented: 16
- Individual tests documented: 123

## `tests/test_bibtex_autofill.py`
- Target: `mcp_research.bibtex_autofill`
- Baseline method: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows.
- Tests: 8

### `test_extract_candidate_signals_prefers_minio_metadata`
- What: Verifies extract candidate signals prefers MinIO metadata.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows.
- Location: `tests/test_bibtex_autofill.py:131`

### `test_enrich_bucket_files_updates_batched_items`
- What: Verifies enrich bucket files updates batched items.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows.
- Location: `tests/test_bibtex_autofill.py:146`

### `test_crossref_resolve_best_match_rejects_low_confidence`
- What: Verifies crossref resolve best match rejects low confidence.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_bibtex_autofill.py:225`

### `test_crossref_resolve_best_match_detects_doi_conflict`
- What: Verifies crossref resolve best match detects DOI conflict.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_bibtex_autofill.py:256`

### `test_enrich_file_metadata_skips_low_confidence_candidates`
- What: Verifies enrich file metadata skips low confidence candidates.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows.
- Location: `tests/test_bibtex_autofill.py:281`

### `test_crossref_book_chapter_maps_to_incollection`
- What: Verifies crossref book chapter maps to incollection.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows.
- Location: `tests/test_bibtex_autofill.py:309`

### `test_normalize_authors_accepts_crossref_style_author_dicts`
- What: Verifies normalize authors accepts crossref style author dicts.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows.
- Location: `tests/test_bibtex_autofill.py:330`

### `test_save_file_metadata_writes_v2_keys_in_v2_mode`
- What: Verifies save file metadata writes v2 keys in v2 mode.
- How: Uses unittest with fake Redis/MinIO/Crossref collaborators and targeted patching to isolate metadata parsing and enrichment flows. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables.
- Location: `tests/test_bibtex_autofill.py:340`

## `tests/test_bibtex_ui_app.py`
- Target: `mcp_research.bibtex_ui_app`
- Baseline method: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence.
- Tests: 27

### `test_index_serves_bibtex_workspace_shell`
- What: Verifies index serves bibtex workspace shell.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:241`

### `test_api_search_status_reports_qdrant_default_collection`
- What: Verifies API search status reports Qdrant default collection.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:250`

### `test_api_search_uses_requested_parameters`
- What: Verifies API search uses requested parameters.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:261`

### `test_api_search_fetch_uses_point_id_and_collection`
- What: Verifies API search fetch uses point ID and collection.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:286`

### `test_api_buckets_lists_minio_buckets`
- What: Verifies API buckets lists MinIO buckets.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:295`

### `test_api_bucket_files_loads_pdf_objects_and_bibtex_metadata`
- What: Verifies API bucket files loads PDF objects and bibtex metadata.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:307`

### `test_api_add_bucket_creates_bucket`
- What: Verifies API add bucket creates bucket.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:339`

### `test_api_delete_bucket_force_removes_objects_and_cleans_ingested_data`
- What: Verifies API delete bucket force removes objects and cleans ingested data.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:349`

### `test_api_delete_bucket_force_preserves_ingested_data_by_default`
- What: Verifies API delete bucket force preserves ingested data by default.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:362`

### `test_api_delete_file_removes_object_and_ingested_data`
- What: Verifies API delete file removes object and ingested data.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:375`

### `test_api_delete_file_preserves_ingested_data_by_default`
- What: Verifies API delete file preserves ingested data by default.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:388`

### `test_api_upload_file_forces_ingest_job_even_when_flag_is_false`
- What: Verifies API upload file forces ingest job even when flag is false.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:401`

### `test_api_upload_multiple_files_queues_celery_ingest_tasks`
- What: Verifies API upload multiple files queues celery ingest tasks.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:417`

### `test_api_upload_rejects_non_pdf_file`
- What: Verifies API upload rejects non PDF file.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:448`

### `test_api_save_file_bibtex_persists_to_redis_prefix`
- What: Verifies API save file bibtex persists to Redis prefix.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:460`

### `test_api_file_redis_summary_counts_partitions_and_chunks`
- What: Verifies API file Redis summary counts partitions and chunks.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:494`

### `test_api_save_file_bibtex_preserves_trailing_spaces_in_title`
- What: Verifies API save file bibtex preserves trailing spaces in title.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:510`

### `test_api_file_redis_data_lazy_limits_items`
- What: Verifies API file Redis data lazy limits items.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:528`

### `test_api_file_redis_summary_reads_v2_schema`
- What: Verifies API file Redis summary reads v2 schema.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:547`

### `test_api_bucket_autofill_missing_processes_in_batches`
- What: Verifies API bucket autofill missing processes in batches.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:569`

### `test_api_file_lookup_by_doi_overwrites_metadata_from_crossref`
- What: Verifies API file lookup by DOI overwrites metadata from crossref.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:633`

### `test_api_file_lookup_by_doi_keeps_existing_authors_when_crossref_has_no_author`
- What: Verifies API file lookup by DOI keeps existing authors when crossref has no author.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:685`

### `test_api_file_lookup_by_doi_uses_parent_doi_authors_for_chapter_doi`
- What: Verifies API file lookup by DOI uses parent DOI authors for chapter DOI.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:729`

### `test_api_file_lookup_by_doi_requires_valid_doi`
- What: Verifies API file lookup by DOI requires valid DOI.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:772`

### `test_api_save_file_bibtex_accepts_extended_entry_types_and_fields`
- What: Verifies API save file bibtex accepts extended entry types and fields.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:786`

### `test_api_file_lookup_by_doi_handles_literal_author_payload`
- What: Verifies API file lookup by DOI handles literal author payload.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:814`

### `test_api_save_file_bibtex_writes_v2_keys_in_v2_mode`
- What: Verifies API save file bibtex writes v2 keys in v2 mode.
- How: Exercises FastAPI routes with TestClient, fake Redis/MinIO/search backends, and mocked integrations to verify API behavior and persistence. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables, is gated by unittest skip decorators.
- Location: `tests/test_bibtex_ui_app.py:847`

## `tests/test_celery_app.py`
- Target: `mcp_research.celery_app`
- Baseline method: Validates environment-driven configuration by overriding env vars and asserting Celery app settings and helper defaults.
- Tests: 2

### `test_env_int_invalid_returns_default`
- What: Verifies environment int invalid returns default.
- How: Validates environment-driven configuration by overriding env vars and asserting Celery app settings and helper defaults. Techniques in this test: overrides environment variables.
- Location: `tests/test_celery_app.py:27`

### `test_make_celery_uses_env_urls`
- What: Verifies make celery uses environment URLs.
- How: Validates environment-driven configuration by overriding env vars and asserting Celery app settings and helper defaults. Techniques in this test: overrides environment variables.
- Location: `tests/test_celery_app.py:32`

## `tests/test_dashboard_app.py`
- Target: `mcp_research.dashboard_app`
- Baseline method: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers.
- Tests: 11

### `test_file_identity_prefers_document_id`
- What: Verifies file identity prefers document ID.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers.
- Location: `tests/test_dashboard_app.py:94`

### `test_scan_collection_files_aggregates_chunks`
- What: Verifies scan collection files aggregates chunks.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers.
- Location: `tests/test_dashboard_app.py:98`

### `test_scan_collection_files_uses_first_page_first_chunk_for_metadata`
- What: Verifies scan collection files uses first page first chunk for metadata.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers.
- Location: `tests/test_dashboard_app.py:118`

### `test_scan_collection_files_dedupes_chunk_and_partition_counts`
- What: Verifies scan collection files dedupes chunk and partition counts.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers.
- Location: `tests/test_dashboard_app.py:147`

### `test_group_chunks_by_partition_in_sequential_order`
- What: Verifies group chunks by partition in sequential order.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers.
- Location: `tests/test_dashboard_app.py:174`

### `test_dedupe_chunk_entries_removes_duplicate_payloads`
- What: Verifies dedupe chunk entries removes duplicate payloads.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers.
- Location: `tests/test_dashboard_app.py:189`

### `test_fetch_file_partition_summaries_omits_chunk_text_payload`
- What: Verifies fetch file partition summaries omits chunk text payload.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_dashboard_app.py:223`

### `test_fetch_partition_chunks_returns_single_partition`
- What: Verifies fetch partition chunks returns single partition.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_dashboard_app.py:255`

### `test_enrich_files_with_redis_counts`
- What: Verifies enrich files with Redis counts.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_dashboard_app.py:288`

### `test_build_original_file_url_from_bucket_key`
- What: Verifies build original file URL from bucket key.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables.
- Location: `tests/test_dashboard_app.py:314`

### `test_attach_original_file_links_uses_source_when_bucket_key_missing`
- What: Verifies attach original file links uses source when bucket key missing.
- How: Uses fake Qdrant/Redis fixtures and deterministic payloads to validate file aggregation, partition grouping, and dashboard enrichment helpers. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables.
- Location: `tests/test_dashboard_app.py:328`

## `tests/test_e2e_integration.py`
- Target: `mcp_research.resolver_app (end-to-end)`
- Baseline method: Runs environment-gated live HTTP checks against the resolver service using real network calls and asserts response contracts.
- Tests: 2

### `test_resolve_doc_json_presign`
- What: Verifies resolve doc JSON presign.
- How: Runs environment-gated live HTTP checks against the resolver service using real network calls and asserts response contracts. Techniques in this test: issues live HTTP requests with httpx, is gated by unittest skip decorators.
- Location: `tests/test_e2e_integration.py:27`

### `test_resolve_doc_redirects`
- What: Verifies resolve doc redirects.
- How: Runs environment-gated live HTTP checks against the resolver service using real network calls and asserts response contracts. Techniques in this test: issues live HTTP requests with httpx, is gated by unittest skip decorators.
- Location: `tests/test_e2e_integration.py:41`

## `tests/test_hybrid_search.py`
- Target: `mcp_research.hybrid_search`
- Baseline method: Combines fake embedding models and fake/live Qdrant clients to verify vector construction, upsert payloads, and hybrid query orchestration.
- Tests: 4

### `test_embed_corpus_builds_dense_and_sparse_vectors`
- What: Verifies embed corpus builds dense and sparse vectors.
- How: Combines fake embedding models and fake/live Qdrant clients to verify vector construction, upsert payloads, and hybrid query orchestration.
- Location: `tests/test_hybrid_search.py:67`

### `test_upsert_docs_sends_payloads_and_vectors`
- What: Verifies upsert docs sends payloads and vectors.
- How: Combines fake embedding models and fake/live Qdrant clients to verify vector construction, upsert payloads, and hybrid query orchestration.
- Location: `tests/test_hybrid_search.py:87`

### `test_hybrid_search_calls_query_points`
- What: Verifies hybrid search calls query points.
- How: Combines fake embedding models and fake/live Qdrant clients to verify vector construction, upsert payloads, and hybrid query orchestration.
- Location: `tests/test_hybrid_search.py:112`

### `test_hybrid_search_hits_live_qdrant`
- What: Verifies hybrid search hits live Qdrant.
- How: Combines fake embedding models and fake/live Qdrant clients to verify vector construction, upsert payloads, and hybrid query orchestration. Techniques in this test: issues live HTTP requests with httpx, is gated by unittest skip decorators.
- Location: `tests/test_hybrid_search.py:147`

## `tests/test_ingest_missing_minio.py`
- Target: `mcp_research.ingest_missing_minio`
- Baseline method: Mocks discovery/enqueue helpers and captures stdout/stderr to validate CLI control flow, dry-run output, and enqueue counts.
- Tests: 3

### `test_print_errors_to_stderr`
- What: Verifies print errors to stderr.
- How: Mocks discovery/enqueue helpers and captures stdout/stderr to validate CLI control flow, dry-run output, and enqueue counts. Techniques in this test: captures CLI output streams.
- Location: `tests/test_ingest_missing_minio.py:22`

### `test_main_dry_run_outputs_missing`
- What: Verifies main dry run outputs missing.
- How: Mocks discovery/enqueue helpers and captures stdout/stderr to validate CLI control flow, dry-run output, and enqueue counts. Techniques in this test: patches collaborators with unittest.mock, captures CLI output streams.
- Location: `tests/test_ingest_missing_minio.py:30`

### `test_main_enqueue_outputs_count`
- What: Verifies main enqueue outputs count.
- How: Mocks discovery/enqueue helpers and captures stdout/stderr to validate CLI control flow, dry-run output, and enqueue counts. Techniques in this test: patches collaborators with unittest.mock, captures CLI output streams.
- Location: `tests/test_ingest_missing_minio.py:45`

## `tests/test_ingest_unstructured.py`
- Target: `mcp_research.ingest_unstructured`
- Baseline method: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding.
- Tests: 9

### `test_load_dotenv_sets_new_values_only`
- What: Verifies load dotenv sets new values only.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding. Techniques in this test: builds temporary filesystem fixtures, overrides environment variables.
- Location: `tests/test_ingest_unstructured.py:70`

### `test_load_env_int_handles_invalid`
- What: Verifies load environment int handles invalid.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding. Techniques in this test: overrides environment variables.
- Location: `tests/test_ingest_unstructured.py:81`

### `test_load_env_bool_parses_true`
- What: Verifies load environment bool parses true.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding. Techniques in this test: overrides environment variables.
- Location: `tests/test_ingest_unstructured.py:86`

### `test_collect_pdfs_from_directory`
- What: Verifies collect PDFs from directory.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding. Techniques in this test: builds temporary filesystem fixtures.
- Location: `tests/test_ingest_unstructured.py:91`

### `test_elements_to_chunks_extracts_page_ranges`
- What: Verifies elements to chunks extracts page ranges.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding.
- Location: `tests/test_ingest_unstructured.py:101`

### `test_upload_to_redis_writes_keys`
- What: Verifies upload to Redis writes keys.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding.
- Location: `tests/test_ingest_unstructured.py:113`

### `test_upload_json_files_to_redis_inferrs_doc_id`
- What: Verifies upload JSON files to Redis inferrs doc ID.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding. Techniques in this test: builds temporary filesystem fixtures.
- Location: `tests/test_ingest_unstructured.py:130`

### `test_main_help_uses_argparse_and_does_not_ingest`
- What: Verifies main help uses argparse and does not ingest.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding. Techniques in this test: patches collaborators with unittest.mock, captures CLI output streams.
- Location: `tests/test_ingest_unstructured.py:153`

### `test_main_forwards_pdf_path_and_data_dir_overrides`
- What: Verifies main forwards PDF path and data dir overrides.
- How: Uses temporary filesystem fixtures, fake Redis storage, and patched entrypoints to verify chunk extraction, Redis writes, and CLI argument forwarding. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_ingest_unstructured.py:163`

## `tests/test_ingestion_tasks.py`
- Target: `mcp_research.ingestion_tasks`
- Baseline method: Patches minio_ingest processors and invokes Celery task wrappers directly to confirm argument pass-through and completion status.
- Tests: 2

### `test_ingest_minio_object_task_calls_processor`
- What: Verifies ingest MinIO object task calls processor.
- How: Patches minio_ingest processors and invokes Celery task wrappers directly to confirm argument pass-through and completion status. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_ingestion_tasks.py:20`

### `test_delete_minio_object_task_calls_processor`
- What: Verifies delete MinIO object task calls processor.
- How: Patches minio_ingest processors and invokes Celery task wrappers directly to confirm argument pass-through and completion status. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_ingestion_tasks.py:28`

## `tests/test_link_resolver.py`
- Target: `mcp_research.link_resolver and mcp_research.resolver_app`
- Baseline method: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient.
- Tests: 18

### `test_build_source_ref_encodes_key_and_pages`
- What: Verifies build source ref encodes key and pages.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient.
- Location: `tests/test_link_resolver.py:34`

### `test_parse_source_ref_from_portal_url`
- What: Verifies parse source ref from portal URL.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient.
- Location: `tests/test_link_resolver.py:48`

### `test_build_citation_url_uses_env_and_path`
- What: Verifies build citation URL uses environment and path.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: overrides environment variables.
- Location: `tests/test_link_resolver.py:63`

### `test_resolve_link_portal_defaults_to_local_base`
- What: Verifies resolve link portal defaults to local base.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: overrides environment variables.
- Location: `tests/test_link_resolver.py:71`

### `test_resolve_link_portal_appends_page_and_highlight_query`
- What: Verifies resolve link portal appends page and highlight query.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: overrides environment variables.
- Location: `tests/test_link_resolver.py:82`

### `test_resolve_link_cdn_builds_url`
- What: Verifies resolve link CDN builds URL.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: overrides environment variables.
- Location: `tests/test_link_resolver.py:100`

### `test_resolve_link_proxy_builds_reverse_proxy_url`
- What: Verifies resolve link proxy builds reverse proxy URL.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: overrides environment variables.
- Location: `tests/test_link_resolver.py:112`

### `test_resolve_link_proxy_uses_relative_path_for_localhost_base`
- What: Verifies resolve link proxy uses relative path for localhost base.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: overrides environment variables.
- Location: `tests/test_link_resolver.py:129`

### `test_resolve_link_presign_uses_minio_client`
- What: Verifies resolve link presign uses MinIO client.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables.
- Location: `tests/test_link_resolver.py:144`

### `test_resolve_link_presign_appends_page_fragment_from_source_ref`
- What: Verifies resolve link presign appends page fragment from source ref.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:169`

### `test_resolve_link_presign_appends_highlight_fragment`
- What: Verifies resolve link presign appends highlight fragment.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:184`

### `test_resolve_doc_redirects`
- What: Verifies resolve doc redirects.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:210`

### `test_resolve_doc_json_returns_payload`
- What: Verifies resolve doc JSON returns payload.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:235`

### `test_resolve_doc_focus_params_render_embed_page_for_presign`
- What: Verifies resolve doc focus params render embed page for presign.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:243`

### `test_resolve_doc_proxy_mode_redirects_to_pdf_proxy`
- What: Verifies resolve doc proxy mode redirects to PDF proxy.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:266`

### `test_resolve_pdf_proxy_returns_pdf_payload`
- What: Verifies resolve PDF proxy returns PDF payload.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:285`

### `test_resolve_pdf_proxy_rejects_non_http_url`
- What: Verifies resolve PDF proxy rejects non HTTP URL.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient.
- Location: `tests/test_link_resolver.py:307`

### `test_resolve_pdf_proxy_with_ref_reads_from_minio`
- What: Verifies resolve PDF proxy with ref reads from MinIO.
- How: Validates URL-building and resolver endpoints by varying environment settings, mocking MinIO presign behavior, and exercising FastAPI routes with TestClient. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_link_resolver.py:311`

## `tests/test_mcp_app.py`
- Target: `mcp_research.mcp_app`
- Baseline method: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping.
- Tests: 16

### `test_pages_to_range_handles_empty`
- What: Verifies pages to range handles empty.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:67`

### `test_source_ref_from_payload_prefers_existing`
- What: Verifies source ref from payload prefers existing.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:71`

### `test_source_ref_from_payload_builds_ref`
- What: Verifies source ref from payload builds ref.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:75`

### `test_coerce_qdrant_offset`
- What: Verifies coerce Qdrant offset.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:80`

### `test_file_identity_prefers_document_id`
- What: Verifies file identity prefers document ID.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:84`

### `test_default_collection_key_uses_prefix`
- What: Verifies default collection key uses prefix.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:88`

### `test_normalize_retrieval_mode`
- What: Verifies normalize retrieval mode.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:91`

### `test_citation_key_from_payload_prefers_payload_value`
- What: Verifies citation key from payload prefers payload value.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:97`

### `test_citation_key_from_payload_falls_back_to_bibtex_metadata`
- What: Verifies citation key from payload falls back to bibtex metadata.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:101`

### `test_highlight_query_from_text_normalizes_chunk_text`
- What: Verifies highlight query from text normalizes chunk text.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:111`

### `test_fetch_document_chunks_uses_meta_chunks_key_override`
- What: Verifies fetch document chunks uses meta chunks key override.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:115`

### `test_fetch_document_chunks_reads_v2`
- What: Verifies fetch document chunks reads v2.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:136`

### `test_fetch_chunk_bibtex_returns_metadata_for_chunk`
- What: Verifies fetch chunk bibtex returns metadata for chunk.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:157`

### `test_fetch_chunk_partition_returns_partition_chunks_for_chunk_page`
- What: Verifies fetch chunk partition returns partition chunks for chunk page.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:171`

### `test_search_include_partition_true_returns_partition_summary`
- What: Verifies search include partition true returns partition summary.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:206`

### `test_search_normalizes_page_range_and_text_for_results`
- What: Verifies search normalizes page range and text for results.
- How: Uses fake Redis/Qdrant clients with patched helper functions to validate internal normalization helpers and MCP tool response shaping. Techniques in this test: patches collaborators with unittest.mock, is gated by unittest skip decorators.
- Location: `tests/test_mcp_app.py:256`

## `tests/test_mcp_cli.py`
- Target: `mcp_cli`
- Baseline method: Patches dynamic module imports and inspects forwarded argv values to verify command dispatch and help behavior.
- Tests: 3

### `test_main_help_exits_zero`
- What: Verifies main help exits zero.
- How: Patches dynamic module imports and inspects forwarded argv values to verify command dispatch and help behavior.
- Location: `tests/test_mcp_cli.py:19`

### `test_dispatch_forwards_arguments_to_module_main`
- What: Verifies dispatch forwards arguments to module main.
- How: Patches dynamic module imports and inspects forwarded argv values to verify command dispatch and help behavior. Techniques in this test: patches collaborators with unittest.mock, simulates CLI argv input.
- Location: `tests/test_mcp_cli.py:24`

### `test_dispatch_minio_ops_command`
- What: Verifies dispatch MinIO ops command.
- How: Patches dynamic module imports and inspects forwarded argv values to verify command dispatch and help behavior. Techniques in this test: patches collaborators with unittest.mock, simulates CLI argv input.
- Location: `tests/test_mcp_cli.py:40`

## `tests/test_minio_ingest.py`
- Target: `mcp_research.minio_ingest`
- Baseline method: Covers ingest helpers with fake Redis/models and patched dependencies to verify event normalization, mapping cleanup, and non-PDF handling.
- Tests: 6

### `test_normalize_events_strips_blanks`
- What: Verifies normalize events strips blanks.
- How: Covers ingest helpers with fake Redis/models and patched dependencies to verify event normalization, mapping cleanup, and non-PDF handling.
- Location: `tests/test_minio_ingest.py:57`

### `test_load_env_list_parses_values`
- What: Verifies load environment list parses values.
- How: Covers ingest helpers with fake Redis/models and patched dependencies to verify event normalization, mapping cleanup, and non-PDF handling. Techniques in this test: overrides environment variables.
- Location: `tests/test_minio_ingest.py:61`

### `test_source_doc_ids_prefers_set_members`
- What: Verifies source doc IDs prefers set members.
- How: Covers ingest helpers with fake Redis/models and patched dependencies to verify event normalization, mapping cleanup, and non-PDF handling. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_minio_ingest.py:65`

### `test_remove_source_mapping_deletes_empty_set`
- What: Verifies remove source mapping deletes empty set.
- How: Covers ingest helpers with fake Redis/models and patched dependencies to verify event normalization, mapping cleanup, and non-PDF handling.
- Location: `tests/test_minio_ingest.py:71`

### `test_delete_from_qdrant_skips_missing_collection`
- What: Verifies delete from Qdrant skips missing collection.
- How: Covers ingest helpers with fake Redis/models and patched dependencies to verify event normalization, mapping cleanup, and non-PDF handling.
- Location: `tests/test_minio_ingest.py:79`

### `test_process_object_skips_non_pdf`
- What: Verifies process object skips non PDF.
- How: Covers ingest helpers with fake Redis/models and patched dependencies to verify event normalization, mapping cleanup, and non-PDF handling.
- Location: `tests/test_minio_ingest.py:88`

## `tests/test_minio_ops.py`
- Target: `mcp_research.minio_ops`
- Baseline method: Uses mocked MinIO clients plus temporary PDF fixtures to validate bucket/file commands and ingest cleanup flags through CLI entrypoints.
- Tests: 4

### `test_add_bucket_creates_when_missing`
- What: Verifies add bucket creates when missing.
- How: Uses mocked MinIO clients plus temporary PDF fixtures to validate bucket/file commands and ingest cleanup flags through CLI entrypoints.
- Location: `tests/test_minio_ops.py:32`

### `test_delete_bucket_force_removes_objects`
- What: Verifies delete bucket force removes objects.
- How: Uses mocked MinIO clients plus temporary PDF fixtures to validate bucket/file commands and ingest cleanup flags through CLI entrypoints.
- Location: `tests/test_minio_ops.py:40`

### `test_upload_file_ingests_by_default`
- What: Verifies upload file ingests by default.
- How: Uses mocked MinIO clients plus temporary PDF fixtures to validate bucket/file commands and ingest cleanup flags through CLI entrypoints. Techniques in this test: patches collaborators with unittest.mock, builds temporary filesystem fixtures.
- Location: `tests/test_minio_ops.py:53`

### `test_remove_file_can_skip_ingested_cleanup`
- What: Verifies remove file can skip ingested cleanup.
- How: Uses mocked MinIO clients plus temporary PDF fixtures to validate bucket/file commands and ingest cleanup flags through CLI entrypoints. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_minio_ops.py:81`

## `tests/test_upload_data_to_redis.py`
- Target: `mcp_research.upload_data_to_redis`
- Baseline method: Checks directory/pair upload flows with temporary JSON fixtures and mocked upload helpers, including strict-mode validation.
- Tests: 3

### `test_upload_pair_calls_helper`
- What: Verifies upload pair calls helper.
- How: Checks directory/pair upload flows with temporary JSON fixtures and mocked upload helpers, including strict-mode validation. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_upload_data_to_redis.py:22`

### `test_upload_directory_requires_pairs_in_strict_mode`
- What: Verifies upload directory requires pairs in strict mode.
- How: Checks directory/pair upload flows with temporary JSON fixtures and mocked upload helpers, including strict-mode validation. Techniques in this test: builds temporary filesystem fixtures.
- Location: `tests/test_upload_data_to_redis.py:43`

### `test_upload_directory_counts_pairs`
- What: Verifies upload directory counts pairs.
- How: Checks directory/pair upload flows with temporary JSON fixtures and mocked upload helpers, including strict-mode validation. Techniques in this test: patches collaborators with unittest.mock, builds temporary filesystem fixtures.
- Location: `tests/test_upload_data_to_redis.py:60`

## `tests/test_upsert_chunks.py`
- Target: `mcp_research.upsert_chunks`
- Baseline method: Uses fake embedder/Qdrant/Redis fixtures and env overrides to verify chunk loading, payload schema, batching, and deterministic ID behavior.
- Tests: 5

### `test_batched_splits_items`
- What: Verifies batched splits items.
- How: Uses fake embedder/Qdrant/Redis fixtures and env overrides to verify chunk loading, payload schema, batching, and deterministic ID behavior.
- Location: `tests/test_upsert_chunks.py:75`

### `test_load_chunk_items_filters_invalid_entries`
- What: Verifies load chunk items filters invalid entries.
- How: Uses fake embedder/Qdrant/Redis fixtures and env overrides to verify chunk loading, payload schema, batching, and deterministic ID behavior. Techniques in this test: builds temporary filesystem fixtures.
- Location: `tests/test_upsert_chunks.py:83`

### `test_load_chunk_items_from_redis`
- What: Verifies load chunk items from Redis.
- How: Uses fake embedder/Qdrant/Redis fixtures and env overrides to verify chunk loading, payload schema, batching, and deterministic ID behavior. Techniques in this test: patches collaborators with unittest.mock.
- Location: `tests/test_upsert_chunks.py:97`

### `test_upsert_items_sends_points`
- What: Verifies upsert items sends points.
- How: Uses fake embedder/Qdrant/Redis fixtures and env overrides to verify chunk loading, payload schema, batching, and deterministic ID behavior.
- Location: `tests/test_upsert_chunks.py:113`

### `test_upsert_items_v2_payload_and_deterministic_id`
- What: Verifies upsert items v2 payload and deterministic ID.
- How: Uses fake embedder/Qdrant/Redis fixtures and env overrides to verify chunk loading, payload schema, batching, and deterministic ID behavior. Techniques in this test: patches collaborators with unittest.mock, overrides environment variables.
- Location: `tests/test_upsert_chunks.py:143`
