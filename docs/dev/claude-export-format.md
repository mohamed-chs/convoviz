# The Claude export format (as of November 2026)

Anthropic documents how to request an export but not what is in it. This page
records what is known about the claude.ai data export, how sure we are, and
what convoviz does about it. Confidence levels are as in
[`export-format.md`](export-format.md): **high** means several independent
sources or published real-export evidence, **med** means one good source or
consistent parser code, **low** means a single secondhand mention.

No real Claude export was used to write convoviz's Claude support. The facts
below come from public evidence, gathered in November 2026:

- **corpus-yoga**'s versioned JSON Schemas of the export
  ([`rsc/schema/pipeline/chat-export/claude/`](https://github.com/rpus/corpus-yoga/tree/main/rsc/schema/pipeline/chat-export/claude)).
  The maintainer validates every export they receive against them, and the
  [conversations changelog](https://github.com/rpus/corpus-yoga/blob/main/rsc/schema/pipeline/chat-export/claude/conversations/CHANGELOG.md)
  dates each change from March to November 2026 (v1 to v24). This is the most
  precise source, but it covers one account, so it lists what that account
  used, not everything that exists.
- First-hand inspections of 2026 exports published in issues:
  [glia#14](https://github.com/Galaxymak1/glia/issues/14) (Oct 22),
  [ClaudeProjectExport#15](https://github.com/pdlourenco/ClaudeProjectExport/issues/15) (Nov 3),
  [corpus-yoga#522](https://github.com/rpus/corpus-yoga/issues/522) (Aug 24),
  [ravel#11](https://github.com/repolex-ai/ravel/issues/11) (Apr 6 export),
  [nexus-ai-chat-importer#63, #72, #73](https://github.com/Superkikim/nexus-ai-chat-importer/issues/73).
- Parsers that read real exports: Nexus AI Chat Importer
  ([format notes](https://github.com/Superkikim/nexus-ai-chat-importer/blob/main/docs/architecture/providers/claude-export-format.md)),
  osteele/claude-chat-viewer (`docs/schemas.md`),
  kadirkatirci/claude-conversation-analyzer (`docs/schema.md`), and about a
  dozen smaller 2026 projects (ctk, llm-anthology, agentsview, Cherry Studio, …).
- Anthropic's help pages:
  [Export your Claude data](https://support.claude.com/en/articles/9450526-export-your-claude-data),
  [Export your organization's data](https://support.claude.com/en/articles/13346720-export-your-organization-s-data).

When you learn something new, update this page with a link to the source.

## Packaging

| Fact | Conf. | convoviz |
|---|---|---|
| Requested from Settings → Privacy → Export data (web or desktop app, not mobile). Free to Max/Ultra plans; Team/Enterprise members only if an owner allows it. A link arrives by email and expires after 24 hours, and may work only once. | high (official) | Documented in the README. |
| Until about August 2026: one ZIP, `data-<account uuid>-<epoch>-<hex>-batch-0000.zip` (earlier `data-YYYY-MM-DD-HH-MM-SS-batch-0000.zip`), holding `conversations.json`, `users.json`, `projects.json` or `projects/<uuid>.json`, `memories.json`, sometimes `login_history.json`. | high | Read like any ZIP. |
| Since about 2026-08-24, the email link downloads `manifest-<account>-<epoch>-<hex>-<datetime>.json` (`version: "1.0"`, `instructions`, `created_at`, `total_files`, `data_files: [{batch_index, export_url, category, part, filename}]`) listing one ZIP per category: `conversations-000.zip`, `projects-000.zip`, `memories-000.zip`, `light_metadata-000.zip`, and on some accounts `frames-000.zip`, `feedback-000.zip`, `design_chats-000.zip`, `claude_code_cloud-000.zip`. Which categories appear varies by account and date range. | high | `conversations-000.zip` (alone, with the other ZIPs, or in a folder with them) is enough. Passing the manifest itself explains what to download. ZIPs without conversations next to one with conversations are ignored. |
| Each category ZIP holds its files at their final paths: `conversations.json`; `projects/<uuid>.json`; `memories/<account>.json`; `users.json` and `login_history.json` (with login IP addresses); `artifacts/<id>/artifact.json` + `versions/…` (published artifacts); `reflections/…`; `claude_code_cloud/routines.json`, `routine_runs.json` (no transcripts). | high | Only `conversations.json` is read. |
| Large categories may be split into `-001`, `-002`, … parts. | low (inferred from the naming) | Parts given together are read together. |
| The export UI offers a date range (all, 30 days, 90 days, custom). | med (Nexus) | Re-running with a newer or overlapping export updates notes by conversation id. |
| Uploaded files are not in the export ("They don't include the original files you uploaded"). Files Claude created (docx, xlsx, pptx, pdf) are not either. | high (official) | See *Files* below. |

## Conversations (`conversations.json`)

A single JSON array, written on one line. 256 MB for 1,463 conversations was
reported. The schema below is corpus-yoga's v24 unless noted.

| Fact | Conf. | convoviz |
|---|---|---|
| Conversation keys: `uuid`, `name`, `summary`, `created_at`, `updated_at`, `account: {uuid}`, `chat_messages`. | high | — |
| `name` is often empty. `summary` is almost always `""`. | med | An empty name falls back to the start of the first prompt. |
| **No project link**: conversations don't say which project they belong to (ravel#18 for April, glia#14 for October). | high | No `project_id` for Claude notes. `project_uuid`/`is_starred`/`model`/`current_leaf_message_uuid` are read if present (some parsers declare them; no 2026 sample has them). |
| Timestamps are ISO 8601 with microseconds, `Z` (2024) or `+00:00` (2026). | high | Parsed as aware datetimes. |
| Message keys: `uuid`, `text`, `content`, `sender` (`human`/`assistant`), `created_at`, `updated_at`, `attachments`, `files`, `parent_message_uuid`. No model per message (API fields `id`, `stop_reason`, `usage`, `model` are stripped). | high | — |
| **Branches are exported**: `parent_message_uuid` links each message to its parent; edits and retries create siblings. Roots point to `00000000-0000-4000-8000-000000000000`. glia#14 found 12 conversations with forks and one with two roots. The field appeared around March–April 2026 (corpus-yoga v3); older exports have none. | high | The tree is rebuilt from parents. Without parents, messages are chained in file order. A parent missing from the conversation attaches to the previous message. |
| No active-leaf pointer in the export (`current_leaf_message_uuid` exists only in the web app's API). | med | The most recent branch is shown, the others are folded at the fork. |
| `text` duplicates the text blocks and is empty on some messages (672 of 3,255 in one sample). Old exports may have `text` and no `content`. | high | `content` wins; `text` is the fallback. |
| Message order in the array is not guaranteed chronological. | med (Nexus 1.6.7) | Order comes from the tree and `created_at`. |
| Variants from browser tools use `messages` instead of `chat_messages` (MemPalace). | low | Both accepted. |

## Content blocks (`chat_messages[].content[]`)

Every block has `type`, `start_timestamp`, `stop_timestamp` and `flags`
(null). Text blocks lost their timestamps after 2026-08-28.

| `type` | Fields | Conf. | Rendered as |
|---|---|---|---|
| `text` | `text`, `citations: [{uuid, start_index, end_index, details: {type: "web_search_citation", url}}]`, `citations_grouping_mode` | high | Prose. Citations become footnotes after `end_index`; titles come from the search results of the same conversation. |
| `thinking` | `thinking`, `summaries: [{summary}]`, `cut_off`, `truncated`, `signature`, `hidden`, `thinking_hidden`, `alternative_display_type` | high | Folded reasoning, titled with the last summary. Empty blocks are skipped. |
| `redacted_thinking` | — | low | Hidden. |
| `tool_use` | `id`, `name`, `input`, `message`, `integration_name`, `icon_name`, `display_content`, `mcp_server_url`, approval fields, `hidden_in_chat` | high | Per tool, below. |
| `tool_result` | `tool_use_id`, `name` (empty for the renamed tools since 2026-10), `content: [items]`, `is_error`, `structured_content`, `meta`, `display_content` | high | Per tool, below. Items: `text` (`text`), `knowledge` (`title`, `url`, `text`, `metadata`), `local_resource` (`file_path`, `name`, `mime_type`), `image` (`file_uuid`). |
| `injected_prompt_block` | `prompt`, `injection_source` (`date_note`, `memory_block_head`, `agent_listing`, `deferred_tools_delta`, `mcp_instructions_delta`, …), since 2026-08-26 | high | Hidden (context claude.ai added, not typed by the user). Broke at least one importer. |
| `voice_note` | `title`, `text` | med | Prose. |
| `token_budget` | `remaining` / `budget` | med | Hidden. |
| `image`, `document` | unknown (counted in glia#14) | low | An image or file "not included in the export" when a name or file id is present, otherwise reported as unknown. |
| anything else | — | — | Its text if it has one, plus a diagnostic. |

## Tools

Tool names change: in October 2026 several were renamed after Claude Code's
(`bash_tool` → `Bash`, `web_search` → `WebSearch`, `web_fetch` → `WebFetch`),
and MCP tools went from `Server:tool` to `mcp__server__tool`.

| Tool | Input | Rendered as |
|---|---|---|
| `artifacts` (2024–2025) | `command` (`create`, `update`, `rewrite`, `view`), `id`, `type` (`application/vnd.ant.code`, `text/markdown`, `text/html`, `image/svg+xml`, `application/vnd.ant.react`, `application/vnd.ant.mermaid`), `title`, `content`, `language`, `old_str`/`new_str`, `version_uuid`; Research reports add `source`, `md_citations` | Artifact block with the full text after each edit, per branch; final version saved under `artifacts/`. |
| `create_file`, `Write` | `path`/`file_path`, `file_text`/`content` | File block, same as artifacts. |
| `str_replace`, `Edit` | `path`/`file_path`, `old_str`/`old_string`, `new_str`/`new_string`, `replace_all` | Applied to the file's latest text. |
| `visualize:show_widget` | `title`, `widget_code` (SVG/HTML) | Widget block (HTML), saved like artifacts. |
| `bash_tool`, `Bash` | `command`, `description` | Folded code + output. |
| `repl` (analysis tool, 2024–2025) | `code` (JavaScript) | Folded code + output. |
| MCP tools (`Server:tool`, `mcp__server__tool`) | anything | Folded call (JSON input) + output. |
| `present_files` | `filepaths`; result items `local_resource` | Each file as "not included in the export" (the bytes never are). |
| `memory_user_edits` (`add`/`remove`/`replace`) | — | "*Memory updated*". `view` is hidden. |
| `launch_extended_search_task` | — | "*Started research*". |
| `web_search`, `WebSearch`, `web_search_fast`, `web_fetch`, `WebFetch`, `view`, `Read`, `conversation_search`, `recent_chats`, `memory_read`, `tool_search`, `ToolSearch`, `search_mcp_registry`, `visualize:read_me`, `ask_user_input_v0`, `TaskCreate`, `TaskUpdate`, `Skill`, `Artifact`, `SuggestConnectors`, `recommend_claude_apps`, … | — | Hidden (counted). Search results still give citation titles. |

## Files

| Fact | Conf. | convoviz |
|---|---|---|
| `attachments: [{file_name, file_size, file_type, extracted_content}]`: pasted text and uploaded documents, as the text Claude extracted. `file_name` is often empty for pastes. | high | A folded block with the text, named after the file. |
| `files: [{file_name, file_uuid}]`: uploaded images and other files, without their bytes. Images have no extracted text at all. | high | "*[name not included in the export]*". A file also listed in `attachments` is shown once. |
| Published artifacts (`frames-000.zip`, `artifacts/<id>/`) are linked from conversations only by URL. | med | Not read. |

## Other categories

| Category | Contents | convoviz |
|---|---|---|
| `projects` | `{uuid, name, description, is_private, is_starter_project, prompt_template, created_at, updated_at, creator, docs: [{uuid, filename, content, created_at}]}` | Not read (no conversation links to a project). |
| `memories` | `memories/<account>.json`: `conversations_memory` (text) and `memory_files: [{path, content, updated_at}]`; earlier `memories.json` | Not read yet. A memory page like *Custom instructions* is a candidate. |
| `light_metadata` | `users.json` (`uuid`, `full_name`, `email_address`, `verified_phone_number`), `login_history.json` | Never read (personal data). |
| `design_chats` | `{title, project: {uuid, name}, messages: [{role, content: {content, attachments}}]}` | Not read. |
| `claude_code_cloud`, `feedback`, `frames` | Routine metadata, feedback, published artifacts | Not read. |
