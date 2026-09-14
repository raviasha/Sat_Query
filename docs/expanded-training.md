# Expanded linear experiment

The expanded notebook selects 4,600 training areas and the same 200 validation and 200 test areas as the original seeded, country-balanced 1,000-area experiment. The original 600 training areas remain included. It reads the completed official archive parts from the education account; it does not download imagery again.

`src/satquery/official_parts.py` verifies ordered parts and presents a sequential stream. `scripts/colab_official_subset.py` checks the original full metadata SHA256, preserves the original selection identities, streams Zstandard TARs, and copies only expected bands to trusted paths. Selected modalities are cached as checksum-verified ZIPs before creating the paired shards expected by existing preprocessing. Archive traversal paths are never used as output paths.

`scripts/build_expanded_notebook.py` packages current code and adapts the existing stage scripts to 5,000 areas. Normalization, CROMA checkpoint, linear architecture, optimizer, seed, validation selection and bootstrap repetitions remain unchanged. Every selected reference pixel and token footprint is audited by the existing independent audit.

Outputs live in `MyDrive/SatQuery/bigearthnet-v2-5000` and `MyDrive/SatQuery/pipeline-5000`. Run `notebooks/SatQuery_Pipeline_5000.ipynb` under the education account. Feature extraction requires a T4 GPU. Archive extraction streams all compressed bytes and can take hours. Completed modality caches and complete pipeline stages can be reused; an interrupted incomplete stage may need recomputation. Archives remain untouched.

The test set has already informed the original baseline report. Treat repeat results as a controlled comparison, not an untouched final benchmark. Model selection uses validation only. Baseline test dominant accuracy was 53.8743%, with coverage MAE 6.1553 percentage points. Report class-specific errors and support alongside overall metrics; predicted coverage is not calibrated confidence.

Status: notebook prepared; expanded model results must only be reported after the live run completes.


## Live run interruption — 11 September 2026

Notebook: https://colab.research.google.com/drive/1Phk8g1QR5TQA-ATQfmmrC63KZ6Cwhqdx?authuser=4

Selection checks passed (4,600/200/200), and all 5,000 reference maps were extracted and saved as a verified modality cache. SAR archive reading consistently fails after verified part 2/203 with OSError 107, Transport endpoint is not connected. A forced Drive remount and then a replacement runtime with fresh authorization both reproduced this failure. About 68 GB of local space was available. Drive service logs contained missing saved-credential errors, but the underlying cause is not established. Do not repeatedly restart the same extraction expecting different results. Investigate Drive read stability or a different archive transport. Official download files were not modified. Tensor preparation, CROMA extraction, training and evaluation have not started.


## Bounded-read diagnostic and fix — 11 September 2026, 01:00 UTC

After reconnecting Drive, reading SAR part 00002.part in 1 MiB chunks succeeded and matched its saved SHA256 (268,435,456 bytes, about 2.8 seconds). This supports testing bounded reads in place of Path.read_bytes(), but full-run stability is not yet established. PartsReader now copies each part in 1 MiB reads into a temporary local spool, verifies the part checksum before exposing data, and retains the full-archive MD5 check. A regression test rejects unbounded part reads; it failed before the fix and passed afterward. All 12 archive-reader and downloader tests passed.

A temporary compatibility patch cell was entered and run in the live notebook immediately AFTER the extraction cell, replacing the diagnostic cell. The Mac locked before its output could be checked. On resumption, verify that cell succeeded, MOVE IT ABOVE the extraction cell so Run all applies the patch before extraction, and then resume the pipeline. The rebuilt local notebook embeds the corrected module directly and does not need this compatibility cell. Browser automation explicitly requested manual unlock; do not bypass that lock. Training remains unstarted.


## Recovery confirmed — 11 September 2026

After the user unlocked the Mac, the live compatibility patch was verified, moved above extraction, and Run all activated. The corrected reader progressed through SAR parts 1–6/203, beyond the previous repeated failure after part 2. Full archive scanning and downstream stages remain in progress; no expanded training results exist yet. The local notebook was rebuilt with the corrected reader. Focused tests: 12 passed; Ruff passed. The live patch remains necessary because that notebook installer embeds the original wheel.


## Monitor check — 11 September 2026, 01:35 UTC

Chrome initially exposed the exact old six-part / Executing (25s) display, so it could not establish fresh progress. Requested a browser page reload (not a runtime restart); Chrome then exposed only an empty window with browser toolbar, with no notebook content even after a bounded wait. No new runtime error or completion was confirmed. Do not treat the cached six-part display as current progress. Next check must restore visible notebook access and inspect actual runtime output before resuming or reporting progress.

Monitor 2026-09-11 02:14 UTC: browser still exposes only an empty window and toolbar; notebook content remains unavailable. No runtime action taken and no fresh progress verified. Waiting for the already-requested visible notebook access; no repeated notification.


## User-requested continuation — 11 September 2026

A fresh tab of the SAME Colab notebook loaded successfully. Saved output confirms SAR parts 203/203, all 10,000 selected SAR TIFFs extracted, and Saved verified selected modality: BigEarthNet-S1. Optical then failed after part 2/236 with ConnectionAbortedError 103 (software caused connection abort) opening BigEarthNet-S2.tar.zst.parts/00002.part. The runtime had ended. Run all was resumed on a new GPU runtime, and Drive mounted successfully; execution was observed. The user then actively switched to another Chrome tab, so further notebook interaction was deferred rather than interfering. Next monitor should inspect the fresh notebook tab (there are two views of the same notebook) for cache reuse and optical progress. No training results yet.

Latest user status check: optical extraction failed again with OSError 107 in PartsReader bounded source.read(1 MiB) / path.open. The current replacement runtime is connected but extraction has stopped. Small reads enabled full SAR extraction but did not eliminate the underlying Drive instability. Preserve both completed caches; investigate optical transport rather than repeating blind restarts. Monitoring changed to every 10 minutes at user request. No training results yet.


## Alternative transport probe — 11 September 2026, 03:39 UTC

A bounded public-source probe succeeded against https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1 : HTTP 206, Content-Range bytes 536870912-536871935/63251710377, exactly 1,024 bytes. This is the offset of the repeatedly failing third optical Drive part. Only a 1 KiB probe was transferred to the Mac; no large archive downloaded. Existing HTTPRanges in src/satquery/remote_lmdb.py validates exact range status, length, and Content-Range and retries transient failures. A next recovery option is streaming optical bytes from this official source in bounded chunks into the existing selective TAR extractor, preserving full archive MD5 2245ed2d1a93f6ce637d839bc856396e and the fixed selected IDs; do not materialize the whole 63 GB archive on the Mac or assume this small probe proves full transfer reliability. Completed SAR and reference-map caches remain reusable. Chrome was on another user task, so no browser mutation was performed during this probe.


## Official HTTP fallback implemented — 11 September 2026, 03:51 UTC

Added src/satquery/archive_http.py (HTTPArchiveReader) using existing strict/retrying HTTPRanges. It buffers at most one 8 MiB range at a time for bounded caller reads, tracks full archive MD5, and only marks finished after EOF checksum succeeds. Three new tests cover boundary/readinto behavior, corrupt data, and short range rejection; tests were red before implementation, then all 7 HTTP/archive reader tests passed and Ruff passed. scripts/colab_official_subset.py now uses HTTPArchiveReader for optical only, pinned official URL/size/MD5, while reusing completed modality caches and recording transport in the cache receipt.

NOT LIVE YET: prepared /tmp/satquery_optical_recovery.py as a standalone Colab-upload script, which mounts Drive, installs the new small module into the existing satquery package, then runs the updated extraction script. It does not need a new runtime/wheel. Upload it through Colab Files, then run it IN PLACE OF the old extraction cell (not after executing the old failing cell). After extraction completes run stages 1–7. The local notebook is being rebuilt with the latest module. Browser file-panel interaction exposed no controls and screenshot was blank, so live upload/execution is unconfirmed. Do not claim fallback running.

## Updated notebook live — 11 September 2026

User uploaded rebuilt notebook and ran all. Active notebook is now https://colab.research.google.com/drive/1TjHTchNodFrXzsDHIIhPJVN5_xgB1nd4?authuser=4 (do not run older notebook). It embeds the HTTP optical fallback and bounded PartsReader; no compatibility patch required. Initial Drive mount failed while authorization was being completed. After completing existing account consent, one retry succeeded: Mounted at /content/drive, fixed 4600/200/200 selection, reused verified Reference_Maps and BigEarthNet-S1 selections. Live output confirms Archive transport: official HTTP ranges and Official HTTP stream: 268435456/63251710377 bytes. This verifies actual optical transfer has started, not full completion. Tensor preparation and training remain pending. Ten-minute monitor updated to the NEW URL and continued through all stages.

Monitor 2026-09-11 04:22 UTC: Chrome is occupied by a user file-open dialog for an unrelated CSV. Did not dismiss it or switch tabs, to avoid interfering with active user work. No fresh notebook progress or failure verified; last verified optical count remains 268,435,456 bytes. Monitor remains active.

Monitor 2026-09-11 04:35 UTC: Chrome foreground is the unrelated Create Fresher Learning Plans page. No browser mutation performed to avoid disrupting user work. No new notebook progress verified; scheduled monitor continues.

User status check: new notebook showed Reconnect GPU. Fullscreen saved optical output reached 7,784,628,224 / 63,251,710,377 bytes and 11,000 / 60,000 selected TIFFs extracted, with no final checksum or completion visible. This is historical progress, not proof runtime is running. During attempted return to runtime controls, CUA reported user changed Chrome; refreshed foreground was Home - Google Drive. Deferred further interaction. Need reconnect existing runtime and inspect status; do not assume runtime terminated or restart extraction blindly.

Monitor 2026-09-11 05:52 UTC: exact setup failure visually confirmed: MessageError: Error: credential propagation was unsuccessful, in drive.mount. Runtime connected but pipeline idle. Attempted one setup retry, but CUA reported user changed Chrome before action completed; retry NOT confirmed. Needs successful Drive credential handoff, then cache reuse and optical scan. Do not claim execution resumed.

Monitor 2026-09-11 06:12 UTC: recovery verified in live notebook: Mounted at /content/drive, reused verified Reference_Maps and BigEarthNet-S1 selections, official HTTP stream 268,435,456 / 63,251,710,377 bytes, extraction executing. Optical scan restarted; this is not cumulative with previous 7.8 GB scan. No tensor/training completion yet. Continue monitoring.

Monitor 2026-09-11 06:28 UTC: live output has advanced to at least 805,306,368 optical bytes and runtime indicates executing. Output prefix is truncated, so actual total may be higher. Attempt to inspect log tail stopped when CUA reported user changed Chrome; no further mutation. No new error or completed stage verified.

Monitor 2026-09-11 07:07 UTC: accessibility exposes truncated optical log prefix through 1,342,177,280 bytes, but full-output scrolling/screenshots remain unavailable (noWindowsAvailable). Cannot establish fresh total or runtime completion from this cached prefix. No restart or other runtime mutation performed. Monitoring remains active; no new failure of the pipeline itself confirmed.

Monitor 2026-09-11 07:30 UTC: Chrome exposes only browser toolbar, no notebook content. Requested page refresh (not runtime restart), but content remains unavailable. No fresh progress verified; monitoring visibility remains blocked.

Monitor 2026-09-11 07:58 UTC: notebook briefly visible with extraction cell stop indicator and Executing status, but output exposes only 268 MB prefix. Cannot establish fresh byte total or continuity; after requesting output menu, browser returned toolbar-only state again. No completion verified and no runtime restart performed.

Monitor 2026-09-11 08:10 UTC: successfully scrolled to actual extraction output on notebook page (not fullscreen). Full visible log ends at 805,306,368 / 63,251,710,377 bytes, with stop indicator on extraction cell and later stages queued. This advanced from 268 MB at 07:58; transfer is progressing slowly. No error displayed. Keep current run, do not restart. Page now positioned at extraction output for easier monitoring.

Monitor 2026-09-11 08:21 UTC: Chrome foreground is user GitHub task with address field focused. Did not switch tabs or interfere. No fresh pipeline output verified; monitor continues.

Monitor 2026-09-11 08:32 UTC: Chrome is on unrelated Render database task. No browser mutation, no fresh notebook progress verified. Monitoring remains active.

Monitor 2026-09-11 08:43 UTC: screenshot confirms current optical scan at 1,879,048,192 / 63,251,710,377 bytes, 1,000 / 60,000 selected TIFFs extracted, extraction stop indicator and subsequent stages queued. No error shown. Slow progress continues; no restart performed.

Monitor 2026-09-11 09:05 UTC: visible log advanced to 2,684,354,560 / 63,251,710,377 bytes and 2,000 / 60,000 selected optical TIFFs. Extraction executing, later stages queued, no error. Preserve running scan.

Monitor 2026-09-11 09:16 UTC: screenshot verifies 3,221,225,472 / 63,251,710,377 optical bytes; latest selected TIFF milestone remains 2,000. Extraction running, subsequent stages queued, no error visible. No intervention.

Monitor 2026-09-11 09:27 UTC: optical scan advanced to 3,489,660,928 / 63,251,710,377 bytes, executing with no visible error. Later stages queued. No intervention.

Monitor 2026-09-11 09:38 UTC: screenshot shows optical 4,026,531,840 / 63,251,710,377 bytes, extraction executing, no visible error. Later pipeline stages remain queued. No intervention.

Monitor 2026-09-11 09:50 UTC: optical scan advanced to 4,563,402,752 / 63,251,710,377 bytes; selected TIFF milestone 3,000 / 60,000. Extraction running, no error shown; later stages queued. No intervention.
