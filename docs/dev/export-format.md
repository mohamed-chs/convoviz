# The ChatGPT export format (as of November 2026)

OpenAI doesn't document this format, and it changes without notice. This page
records what is known, how sure we are, and what convoviz does about it.
Confidence: **high** means several independent sources or real-export
evidence, **med** means one good source or consistent parser code, **low**
means a single secondhand mention.

When you learn something new, update this page, preferably with a link to a
public source. `convoviz inspect --json` output from users is the best evidence
of all.

## Packaging

| Fact | Conf. | convoviz |
|---|---|---|
| Delivered as a ZIP by email (Settings → Data controls → Export data). | high | `source.py` reads ZIPs in place. |
| Large exports split conversations into `conversations-000.json`, `-001`, …; small ones still use a single `conversations.json`. | high | Both handled. Shards are read in numeric order. |
| `export_manifest.json`: `{"version": 1, "logical_files": {"conversations.json": {"files": [...], "sharded": true, "shard_count": N}}}`. | high | It is used first. Missing shards are reported as errors, and a count mismatch as a warning. |
| Some exports may have a manifest with an inventory instead: `{"export_files": [{"path", "size_bytes"}, …]}` ([difegam](https://difegam.github.io/chatgpt-to-markdown/explanation/export-format)). | low | Conversation shards are taken from the inventory, and missing ones are reported. |
| Sharding started around mid-February 2026 ([forum, Mar 24](https://community.openai.com/t/decoding-exported-data-by-parsing-conversations-json-and-or-chat-html/403144?page=2)). | med | — |
| Some downloads are several ZIPs: `…-part-0001.zip` holds the conversations, later parts hold attachments. | med | Sibling parts are opened together. |
| Privacy Portal requests yield a ZIP containing `Conversations__….zip` and `Files__….zip`. | med | ZIPs are unpacked (depth ≤ 2), but only when no conversation data is visible, so uploaded ZIP attachments are left alone. |
| Other top-level files: `chat.html`, `user.json`, `user_settings.json`, `message_feedback.json`, `shared_conversations.json`, `library_files.json`, `conversation_asset_file_names.json`, sometimes `sectioned_conversations.json` (`[]`), `group_chats.json`, `ads.json`, a `sites/` folder. | high (presence), med (meaning) | Only the conversation files, the asset name map and the library are read. |
| Business/Enterprise/Edu workspaces cannot export. | high | Documented in the README. |

## Conversations

| Fact | Conf. | convoviz |
|---|---|---|
| A shard is a JSON array of conversations. Older tools also saw `{"conversations": [...]}` or a bare object. | high | All three are accepted. |
| Arrays can contain `{}` or other junk entries. | high | Skipped and counted. |
| Keys: `id`, `conversation_id`, `title`, `create_time`, `update_time` (epoch seconds, float), `mapping`, `current_node`, `default_model_slug`, `gizmo_id`, `is_archived`, `is_starred`, … | high | `gizmo_id` starting with `g-p-` is a Project, otherwise a custom GPT. |
| **Nodes no longer list `children`**, only `parent`. | high | The tree is always rebuilt from `parent`. Legacy `children` is only a hint for nodes whose parent pointer is missing. |
| `current_node` is the leaf the user last viewed, and it may be missing or dangling. | high | Falls back to the most recent leaf. |
| Messages may lack `status`, `weight`, `create_time`, `metadata`. | high | Everything is optional. Times also accept milliseconds and ISO strings. |
| `parts` items are strings, `{content_type: "text", text}` objects, or asset pointers. | high | Handled. |
| Interrupted final replies can be empty. | high | Empty messages are dropped. |
| Strings may contain unpaired UTF-16 surrogates (emoji split in two). | low | orjson is tried first, then a `surrogatepass` fallback re-pairs them (with a warning). |

## Messages convoviz hides or transforms

| Kind | What it is | Rendered as |
|---|---|---|
| `author.role == "system"`, `is_visually_hidden_from_conversation` | Prompt scaffolding | hidden |
| message `channel` (`commentary`, `final`, reported in 2026) | GPT-5-style progress notes vs the answer | not used yet; `inspect` reports `role:channel` counts so we can decide |
| `user_editable_context`, `model_editable_context`, `about_user_message`/`about_model_message` | Custom instructions, memory context | hidden (captured in `Conversation.extra`) |
| `tether_browsing_display`, `tether_browsing_code`, `sonic_webpage`, `system_error` | Browsing plumbing | hidden |
| `thoughts` | Reasoning summary (`thoughts[].summary/content`) | folded "Reasoning" |
| `reasoning_recap` | "Thought for 12s" | the title of the reasoning block |
| assistant → `python` / `container.exec` / `jupyter` | Code run by the model | folded code |
| tool `execution_output` (+ `metadata.aggregate_result.messages[].image_url`) | Output and charts | folded output, charts as images |
| assistant → `canmore.create_textdoc` | Canvas creation (`{name, type, content}`) | open "Canvas" block |
| assistant → `canmore.update_textdoc` | Canvas edit: `{updates: [{pattern, replacement, multiple}]}`. The regex dialect is unconfirmed (Python `re` with `\1`, or ECMAScript with `$1`) | edits are replayed on that branch's document with the flags both dialects share, both reference spellings are understood, and the full text is shown |
| tool `multimodal_text` | Generated images (`image_asset_pointer`, prompt in `metadata.dalle.prompt` or `generation.prompt`) | images, status text dropped |
| `tether_quote` | Quoted page | blockquote with a link |
| tool `bio` | Memory written | "*Memory updated*" |
| other assistant → tool calls (`web.run`, `research_kickoff_tool`, `image_gen`, …) | Plumbing | hidden, counted |
| unknown content types or parts | — | generic text, plus a diagnostic |

## Citations (high)

Assistant text contains private-use markers:

```
U+E200 KIND U+E202 PAYLOAD (U+E202 PAYLOAD)* U+E201
```

`KIND` is one of `cite` (payloads like `turn0search3`, `turn2news1`, with
optional `L10-L20` line ranges), `filecite`, `entity` (JSON
`["type","name",…]`), `url`, `products`, `navlist`, `image_group`, `genui`,
`map`, … `U+E203`/`U+E204` wrap highlighted spans, and only the characters are
removed.

`metadata.content_references[]` describes the markers positionally:
`type`, `matched_text`, `start_idx`/`end_idx`, `alt`, and
`items`/`sources`/`fallback_items` (each with `url`, `title`, `attribution`,
`supporting_websites`). Indices are sometimes code points and sometimes UTF-16
units, so convoviz checks that the slice equals `matched_text`, tries UTF-16,
and finally searches for the literal text. Unresolved `cite` markers are looked
up in `metadata.search_result_groups[].entries[].ref_id`. Older messages use
`metadata.citations[]` (`start_ix`/`end_ix`) with `【11†source】` placeholders.
Leftover `citeturn0search1` residue is removed, but the word "cite" is never
stripped. `utm_source=chatgpt.com` is removed from links.

## Attachments

| Fact | Conf. | convoviz |
|---|---|---|
| Messages point at files with `sediment://file_<hex>` (current) or `file-service://file-<base62>` (older). | high | `AssetIndex.resolve` |
| Files sit at the ZIP root as `file_<hex>-<suffix>.dat` / `file-<id>-<name>.dat` with no real extension. Older exports used `dalle-generations/` and `user-<id>/` folders. | high | Indexed by file-id prefix. Root files win over folder copies. |
| `conversation_asset_file_names.json` maps stored name → original name, a `dalle-generations/<uuid>.webp` path, a bare UUID, or `<conv>/audio/<uuid>.wav`. | high | Meaningful names are kept. Generated ones contribute only their extension. |
| The library file is also reported as `library-files.json` ([forum, Aug 26](https://community.openai.com/t/missing-library-files-after-chatgpt-data-export/1392791)). | low | Top-level sidecar files are matched ignoring case and `-`/`_`. |
| `library_files.json` lists `file_id`, `file_name`, `mime_type`, `library_artifact_type`, `origination_thread_id`, `image_gen_generation_id`, `trashed_at`/`deleted_at`. | med (fields), low (`origination_thread_id` = conversation id) | Names and MIME types are used. Entries no message shows are added to the conversation named by `origination_thread_id`. Trashed or deleted entries are ignored. |
| Uploaded files also appear in the user message's `metadata.attachments[]` (`id`, `name`, `mime_type`). | high | Shown as links (images inline). |
| Generated images and library files are **sometimes missing** from exports (an OpenAI-side problem in mid-2026; 97 missing files were reported in Aug–Oct 2026). | med | A placeholder in the note, counted in the summary. |
| Voice: `real_time_user_audio_video_asset_pointer` wraps audio/video pointers, and transcripts are `audio_transcription` parts. | med | Transcript as text, audio as a link. |

File extensions come from the original name, then the stored name, then the
MIME type, and finally the file's first bytes (PNG, JPEG, GIF, WebP, WAV, PDF,
MP3, MP4, Ogg, WebM, …).

## Sources

Primary, all from 2026:
[Loore PR #532](https://github.com/hrosspet/write-or-perish/pull/532) (Oct 20),
[djui/lifelog PR #28](https://github.com/djui/lifelog/pull/28) (Oct 31, 58-part real export),
[wenlan PR #1022](https://github.com/wenlan-app/wenlan/pull/1022) (Oct 28, manifest validation),
[chatgpt-export-reader #3](https://github.com/marelucent/chatgpt-export-reader/issues/3) (Oct 24–26, no `children`),
[Nexus AI Chat Importer #62](https://github.com/Superkikim/nexus-ai-chat-importer/issues/62) and its
[format notes](https://github.com/Superkikim/nexus-ai-chat-importer/blob/main/docs/architecture/providers/chatgpt-export-format.md) (May–Aug, `.dat` assets, library files),
[OpenAI community: missing library files](https://community.openai.com/t/missing-library-files-after-chatgpt-data-export/1392791) (Aug–Oct),
[The Register](https://www.theregister.com/ai-and-ml/2026/07/23/openai-wont-let-some-customers-export-their-chats-but-this-tool-will/5277795) (Jul 23, workspace exports).
Earlier convoviz issues #60, #62, #64, #69 and #70 describe the same changes
from users' side.
