#!/usr/bin/env python3
"""
Mamba-3-Lite Documentation Generator
Converts project markdown files into a responsive, beautifully-styled HTML documentation portal
with full LaTeX math (KaTeX) and syntax highlighting support.
Output directory: docs_html/ (ignored by git).

Design system: the "printed reader" — cool white paper, blue-black ink, one
ink-blue accent, Literata for prose and IBM Plex Mono for code and measured
values. Shares its lineage with the sibling LLaMA-3-Lite portal. See
`assets/style.css`.
"""
import os
import re
import html
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_DIR / "docs_html"

# The docs site is published from this branch (GitHub Pages + CI). Asset and
# blob links must resolve against it even when built from a worktree.
DEFAULT_PUBLISH_BRANCH = "main"

# The book, in reading order. Each entry is one chapter:
#   (relative_path, part, chapter_title, card_blurb)
# Order matters: it is the reading order for the sidebar, the index page and
# the previous/next links at the foot of every chapter.
DOC_FILES = [
    # Part I — Foundations
    ("README.md", "Part I: Foundations", "Overview",
     "The project in one page: a ~434M-parameter Mamba-3 reproduction, "
     "complex-valued SSD state spaces, and how to run it."),
    ("docs/README.md", "Part I: Foundations", "How to read this book",
     "What each chapter covers, where to start, and the map from source file "
     "to chapter."),
    ("docs/concepts/state-space-foundations.md", "Part I: Foundations",
     "SSD foundations",
     "From RNNs to SSMs: the linear recurrence, ZOH discretization, and the "
     "S4 → Mamba-3 arc."),
    ("docs/concepts/ssd-theory.md", "Part I: Foundations",
     "SSD theory",
     "State-space duality, complex states (N-halving), and the chunkwise "
     "algorithm einsum by einsum."),
    ("docs/concepts/mimo.md", "Part I: Foundations", "MIMO head mixing",
     "SISO → MIMO fully-connected mixer across SSM heads — why, the math, "
     "identity init."),
    ("docs/concepts/block-and-stability.md", "Part I: Foundations",
     "Block anatomy and stability",
     "RMSNorm → SSD → MIMO → SwiGLU wiring, every dtype and precision "
     "choice, and the scaling numbers."),

    # Part II — The data
    ("docs/training.md", "Part II: The data", "The data pipeline",
     "Corpus mix, shard format, PretrainDataset layouts, and the data path "
     "end to end."),
    ("docs/guides/quickstart.md", "Part II: The data", "Quickstart",
     "From zero to a running training loop — install, verify the math, full "
     "run, resume."),

    # Part III — The training run
    ("docs/guides/pretrain-cli.md", "Part III: The training run",
     "The pretrain CLI",
     "Every training/pretrain.py:TrainingConfig flag and the optimizer and "
     "scheduler wiring."),
    ("docs/guides/training-runbook.md", "Part III: The training run",
     "The A100 runbook",
     "Launch, monitor, NaN recovery, and resume for a production pre-training "
     "run."),
    ("docs/guides/tuning.md", "Part III: The training run", "Tuning",
     "chunk_size, lr, batch geometry, compile, Triton dispatch — measure, "
     "don't guess."),
    ("docs/guides/extending.md", "Part III: The training run", "Extending",
     "Add an SSM variant or a sanctioned Triton kernel with the test and doc "
     "contract."),

    # Part IV — Reference
    ("docs/references/config-reference.md", "Part IV: Reference",
     "Config reference",
     "ModelConfig fields, the annotated YAML, and the 16.78B-vs-8.0B token "
     "arithmetic."),
    ("docs/references/ssd-reference.md", "Part IV: Reference",
     "SSD reference",
     "The complex scan API (ssd_complex_chunkwise, ssd_naive_complex) and the "
     "kernel contract."),
    ("docs/references/model-reference.md", "Part IV: Reference",
     "Model reference",
     "Mamba3Transformer, Mamba3Block + dispatch, and MIMO — shapes and "
     "wiring."),
    ("docs/references/training-reference.md", "Part IV: Reference",
     "Training reference",
     "PretrainDataset, CheckpointManager, TrainingLogger, and the data "
     "pipeline shim."),
]

PUBLISHED_MD = {rel for rel, *_ in DOC_FILES}

# Reading face for prose, mono for code and measured values.
FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?'
             'family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,500;'
             '0,7..72,600;0,7..72,700;1,7..72,400'
             '&family=IBM+Plex+Mono:ital,opsz,wght@0,400;0,500;0,600;0,700;1,400'
             '&display=swap" rel="stylesheet">')
BOOT_OVERLAY_HTML = (
    '<div id="boot-overlay" aria-hidden="true">'
    '<div class="boot-inner">'
    '<div class="boot-wordmark">Mamba-3-Lite</div>'
    '<div class="boot-line">Loading the notes'
    ' <span class="boot-bar"></span>'
    '</div></div></div>'
)
BOOT_SCRIPT = """<script>
(function () {
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) document.documentElement.classList.add('booting');
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(function () {
            document.documentElement.classList.remove('booting');
            var ov = document.getElementById('boot-overlay');
            if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
        }, reduced ? 0 : 350);
    });
})();
</script>"""

# Shared <head> for every generated page. Doc pages pass highlight.js +
# KaTeX in `extra_head`; the index portal passes "".
HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Fonts — Literata for prose, IBM Plex Mono for code and data. -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    {font_link}
    {boot_script}
{extra_head}
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
</head>
"""

DOC_EXTRA_HEAD = """    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>"""

# Multi-Scale Attention Sinks & YaRN 128K Phase-Space (FIG · A0) — the
# index hero instrument. Sliding-window attention score heatmap with the
# Token 0..3 sink column glowing olive, the W=128 SWA diagonal band,
# 12 global YaRN sampling arcs, and a moving query token scanline.
# The live interactive canvas simulation lives in assets/portal.js.

# 12-Layer Alternating Pass & Top-2 MoE Grouped-GEMM Pipeline (FIG · A1) —
# one full training step. Visualizes the entire forward activation pass,
# autograd backward pass, gradient checkpointing re-computation, and the
# AdamW weight update step across 6 architectural stations (Embed → SWA →
# Global YaRN → MoE Router → Triton GEMM → Chunked CE + AdamW).
# The live interactive canvas simulation lives in assets/portal.js.

def slugify(text: str) -> str:
    """Generate clean HTML id for headings.

    Matches the anchor convention the docs were authored against (GitHub-style):
    lowercase, keep word chars + underscore + hyphen, drop everything else,
    and turn each space into a single hyphen (no run collapsing).
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s', '-', text)
    return text.strip('-') or "heading"


@lru_cache(maxsize=1)
def github_base_url() -> str:
    """Derive the GitHub blob base (https://github.com/<owner>/<repo>/blob/<branch>)."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
        out = out.replace("git@github.com:", "https://github.com/").removesuffix(".git")
        if not out.startswith("https://github.com/"):
            return ""
    except Exception:
        return ""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
    except Exception:
        return ""
    return f"{out}/blob/{branch}" if branch else ""


def github_raw_base_url() -> str:
    """Raw-content twin of `github_base_url`, for <img src> that must render
    rather than open an HTML blob page."""
    blob_base = github_base_url()
    if not blob_base:
        return ""
    return blob_base.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")


# Tags the portal renders from raw markdown HTML. Anything outside this set is
# dropped rather than passed through, so a doc can never inject script.
RAW_ALLOWED_TAGS = {
    'div', 'p', 'span', 'a', 'img', 'br', 'hr', 'em', 'strong', 'b', 'i',
    'code', 'pre', 'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'th',
    'td', 'details', 'summary', 'figure', 'figcaption', 'sup', 'sub', 'kbd',
    'small', 'abbr', 'del', 'ins', 'u',
}
# Attributes kept per tag. `style` is deliberately absent: raw inline styles
# would override the design system's tokens.
RAW_ALLOWED_ATTRS = {
    '*': {'class', 'id', 'title', 'align', 'width', 'height', 'colspan', 'rowspan'},
    'a': {'href', 'target', 'rel'},
    'img': {'src', 'alt', 'loading', 'decoding'},
    'td': {'align'},
    'th': {'align', 'scope'},
    'abbr': {'title'},
}
RAW_VOID_TAGS = {'br', 'hr', 'img'}
_RAW_TAG_RE = re.compile(r'<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9-]*)((?:\s+[^<>]*?)?)\s*(/?)\s*>')


def resolve_asset_src(src: str, src_rel_path: str) -> str:
    """Point an <img src> at something that actually renders.

    The build ships no PNGs, so a repo-relative path becomes a raw GitHub URL.
    Absolute URLs and data URIs pass through untouched.
    """
    src = src.strip()
    if not src or src.startswith(("http://", "https://", "data:", "//", "#")):
        return src
    raw_base = github_raw_base_url()
    if not raw_base:
        return src
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent
    candidate = (src_dir / src).resolve()
    try:
        repo_rel = candidate.relative_to(WORKSPACE_DIR).as_posix()
    except ValueError:
        return src
    return f"{raw_base}/{repo_rel}"


def resolve_raw_href(href: str, src_rel_path: str) -> str:
    """Resolve an href inside a raw HTML island (fix_md_links never sees these)."""
    href = href.strip()
    if not href or href.startswith(("http://", "https://", "mailto:", "#")):
        return href
    path_part, _, anchor = href.partition("#")
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent
    if path_part.endswith(".md"):
        for candidate in (src_dir / path_part, WORKSPACE_DIR / path_part):
            if candidate.exists():
                try:
                    repo_rel = candidate.resolve().relative_to(WORKSPACE_DIR)
                except ValueError:
                    continue
                rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
                return rel[:-3] + ".html" + (f"#{anchor}" if anchor else "")
    repo_base = github_base_url()
    if repo_base:
        try:
            repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
            return f"{repo_base}/{repo_rel}" + (f"#{anchor}" if anchor else "")
        except ValueError:
            pass
    return href


def sanitize_raw_html(raw: str, src_rel_path: str, format_text=None) -> str:
    """Filter a raw HTML island down to the portal's tag/attribute allowlist.

    Relative `src`/`href` are rewritten so they resolve from `docs_html/`,
    where the referenced images and sibling docs do not exist on disk.
    `format_text` formats the text nodes between tags, so markdown emphasis
    inside a passthrough island still renders.
    """
    out = []
    pos = 0
    for m in _RAW_TAG_RE.finditer(raw):
        chunk = raw[pos:m.start()]
        out.append(format_text(chunk) if format_text else html.escape(chunk, quote=False))
        pos = m.end()

        closing, tag, attrs_raw, self_closing = m.group(1), m.group(2).lower(), m.group(3) or '', m.group(4)
        if tag not in RAW_ALLOWED_TAGS:
            continue
        if closing:
            out.append(f'</{tag}>')
            continue

        allowed = RAW_ALLOWED_ATTRS['*'] | RAW_ALLOWED_ATTRS.get(tag, set())
        kept = []
        for name, value in re.findall(r'([a-zA-Z-]+)\s*=\s*"([^"]*)"', attrs_raw):
            name = name.lower()
            if name not in allowed:
                continue
            if tag == 'img' and name == 'src':
                value = resolve_asset_src(html.unescape(value), src_rel_path)
            elif tag == 'a' and name == 'href':
                value = resolve_raw_href(html.unescape(value), src_rel_path)
                if 'target' in attrs_raw.lower() and 'rel' not in attrs_raw.lower():
                    kept.append(('rel', 'noopener noreferrer'))
            elif tag == 'a' and name == 'target' and value not in ('_blank', '_self'):
                continue
            kept.append((name, html.escape(value, quote=True)))

        attr_str = ''.join(f' {n}="{v}"' for n, v in kept)
        if tag not in RAW_VOID_TAGS and self_closing:
            out.append(f'<{tag}{attr_str}></{tag}>')
        else:
            out.append(f'<{tag}{attr_str}>')
    tail = raw[pos:]
    out.append(format_text(tail) if format_text else html.escape(tail, quote=False))
    return ''.join(out)


def looks_like_raw_html_block(text: str) -> bool:
    """True when a paragraph's opening line is a real HTML island, not prose.

    Prose like "a < b and c > d" must stay escaped text; a line whose first
    tag is allowlisted HTML is a passthrough block.
    """
    stripped = text.strip()
    if not stripped.startswith('<'):
        return False
    m = _RAW_TAG_RE.match(stripped)
    if not m:
        return False
    return m.group(2).lower() in RAW_ALLOWED_TAGS


def fix_md_links(content: str, src_rel_path: str) -> str:
    """Rewrite relative markdown links for the HTML build.

    - ``.md`` links (with optional ``#anchor``) -> ``.html`` twin.
    - non-``.md`` repo-relative links -> GitHub blob URL (the file isn't
      shipped inside ``docs_html/``).
    """
    repo_base = github_base_url()
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent

    def link_replacer(match):
        label = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return f"[{label}]({url})"
        path_part, _, anchor = url.partition("#")
        if path_part.endswith(".md"):
            # Resolve the target to a repo-relative .md path. Docs mix two
            # styles: repo-root-relative (``docs/...``) and file-relative
            # (``../...`` / ``basename.md``), so try both and keep the hit.
            if path_part.startswith("/"):
                repo_rel = Path(path_part[1:])
            else:
                cand_src = src_dir / path_part
                cand_root = WORKSPACE_DIR / path_part
                if cand_src.exists():
                    repo_rel = cand_src.resolve().relative_to(WORKSPACE_DIR)
                elif cand_root.exists():
                    repo_rel = cand_root
                else:
                    # Prefer src-relative; falls through to a relative href
                    # the build still ships (harmless if unresolved).
                    repo_rel = Path(src_rel_path).parent / path_part
            repo_rel_posix = repo_rel.as_posix()
            if repo_rel_posix not in PUBLISHED_MD:
                # Not a chapter: point at the source file rather than at a
                # page the build never generates.
                if repo_base:
                    return f"[{label}]({repo_base}/{repo_rel_posix})"
                return f"[{label}]({url})"
            # Emit a href relative to the current output file's directory.
            rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
            target = rel[:-3] + ".html"
            if anchor:
                target += "#" + anchor
            return f"[{label}]({target})"
        if repo_base and not path_part.startswith("/"):
            repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
            return f"[{label}]({repo_base}/{repo_rel})"
        return f"[{label}]({url})"

    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)


def parse_markdown_to_html(md_text: str, src_rel_path: str) -> tuple[str, list[dict]]:
    """Statically convert markdown to rich HTML structure with full LaTeX & Math protection.

    Returns (html_content, toc_items).
    """
    md_text = fix_md_links(md_text, src_rel_path)

    # STEP 1: Protect Code Blocks & Inline Code
    code_blocks = []
    def store_code_block(m):
        code_blocks.append(m.group(0))
        return f"\n\n___CODEBLOCK_{len(code_blocks)-1}___\n\n"

    md_text = re.sub(r'```[\s\S]*?```', store_code_block, md_text)

    inline_codes = []
    def store_inline_code(m):
        inline_codes.append(m.group(0))
        return f"___INLINECODE_{len(inline_codes)-1}___"

    md_text = re.sub(r'`[^`\n]+`', store_inline_code, md_text)

    # STEP 2: Protect LaTeX Math Blocks & Inline Math
    display_maths = []
    def store_display_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        display_maths.append(f'<div class="math-block">$$\n{safe_math}\n$$</div>')
        return f"\n\n___DISPLAYMATH_{len(display_maths)-1}___\n\n"

    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', store_display_math, md_text)
    md_text = re.sub(r'\\\[([\s\S]+?)\\\]', store_display_math, md_text)

    inline_maths = []
    def store_inline_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        inline_maths.append(f'<span class="math-inline">${safe_math}$</span>')
        return f"___INLINEMATH_{len(inline_maths)-1}___"

    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', store_inline_math, md_text)
    md_text = re.sub(r'\\\(([\s\S]+?)\\\)', store_inline_math, md_text)

    # STEP 3: Parse Document Structure Line by Line
    toc = []
    seen_slugs = {}
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    table_headers = []
    table_rows = []

    list_stack = []  # [{'indent': int, 'tag': str, 'li_open': bool}]
    h1_seen = False  # the first H1 duplicates the page's doc-title; suppressed

    in_blockquote = False
    blockquote_type = "normal"
    blockquote_lines = []

    def close_li():
        nonlocal list_stack
        if list_stack and list_stack[-1]['li_open']:
            html_lines.append("</li>")
            list_stack[-1]['li_open'] = False

    def flush_list():
        nonlocal list_stack
        while list_stack:
            close_li()
            html_lines.append(f"</{list_stack[-1]['tag']}>")
            list_stack.pop()

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_type, blockquote_lines
        if in_blockquote:
            content = "<br>".join(blockquote_lines)
            if blockquote_type != "normal":
                title = blockquote_type.upper()
                icon = {"NOTE": "ℹ️", "TIP": "💡", "IMPORTANT": "📌", "WARNING": "⚠️", "CAUTION": "🚨"}.get(title, "ℹ️")
                html_lines.append(
                    f'<div class="callout callout-{blockquote_type.lower()}">'
                    f'<div class="callout-header"><span class="callout-icon">{icon}</span><span class="callout-title">{title}</span></div>'
                    f'<div class="callout-body">{content}</div>'
                    f'</div>'
                )
            else:
                html_lines.append(f'<blockquote>{content}</blockquote>')
            in_blockquote = False
            blockquote_type = "normal"
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if in_table:
            th_html = "".join(f"<th>{h}</th>" for h in table_headers)
            tr_html = ""
            for row in table_rows:
                td_html = "".join(f"<td>{c}</td>" for c in row)
                tr_html += f"<tr>{td_html}</tr>"
            html_lines.append(
                f'<div class="table-container"><table class="doc-table">'
                f'<thead><tr>{th_html}</tr></thead>'
                f'<tbody>{tr_html}</tbody>'
                f'</table></div>'
            )
            in_table = False
            table_headers = []
            table_rows = []

    def escape_preserving_entities(text: str) -> str:
        """Escape `<`, `>` and stray `&`, but leave valid HTML entities intact."""
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        return text.replace('<', '&lt;').replace('>', '&gt;')

    def render_inline_formatting(text: str) -> str:
        text = escape_preserving_entities(text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("___DISPLAYMATH_") or stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if not stripped:
            flush_table()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        # Blockquote or Callout
        if stripped.startswith(">"):
            flush_table()
            flush_list()
            bq_content = stripped.lstrip(">").strip()
            callout_match = re.match(r'^\[\!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', bq_content, re.IGNORECASE)
            if callout_match:
                in_blockquote = True
                blockquote_type = callout_match.group(1).upper()
                remaining = bq_content[callout_match.end():].strip()
                if remaining:
                    blockquote_lines.append(render_inline_formatting(remaining))
            else:
                if not in_blockquote:
                    in_blockquote = True
                    blockquote_type = "normal"
                if bq_content:
                    blockquote_lines.append(render_inline_formatting(bq_content))
            i += 1
            continue

        # Horizontal Rule
        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append("<hr class='doc-hr'>")
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()

            clean_title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text_raw)
            clean_title = re.sub(r'`([^`]+)`', r'\1', clean_title)
            clean_title = re.sub(r'___INLINECODE_\d+___', '', clean_title)
            clean_title = re.sub(r'___INLINEMATH_\d+___', '', clean_title)
            raw_slug = slugify(clean_title)
            if raw_slug in seen_slugs:
                seen_slugs[raw_slug] += 1
                heading_id = f"{raw_slug}-{seen_slugs[raw_slug]}"
            else:
                seen_slugs[raw_slug] = 0
                heading_id = raw_slug
            rendered_heading = render_inline_formatting(heading_text_raw)

            if level in (2, 3):
                toc.append({'level': level, 'title': clean_title, 'id': heading_id})

            if level == 1 and not h1_seen:
                h1_seen = True
                html_lines.append(f'<span class="doc-anchor" id="{heading_id}"></span>')
            else:
                html_lines.append(
                    f'<h{level} id="{heading_id}" class="heading-anchor">'
                    f'{rendered_heading}'
                    f'<a href="#{heading_id}" class="anchor-link" aria-label="Link to section">#</a>'
                    f'</h{level}>'
                )
            i += 1
            continue

        # Markdown Table Detection
        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?---', lines[i + 1].strip()):
            flush_list()
            flush_blockquote()
            in_table = True
            headers_raw = [c.strip() for c in line.strip().strip("|").split("|")]
            table_headers = [render_inline_formatting(h) for h in headers_raw]
            i += 2

            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        # Lists
        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            tag = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            indent = len(line) - len(line.lstrip(' '))

            while list_stack and indent < list_stack[-1]['indent']:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if list_stack and list_stack[-1]['indent'] == indent and list_stack[-1]['tag'] != tag:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if not list_stack or list_stack[-1]['indent'] != indent:
                list_stack.append({'indent': indent, 'tag': tag, 'li_open': False})
                html_lines.append(f'<{tag} class="doc-list">')
            else:
                close_li()

            task_match = re.match(r'^\[([ xX])\]\s+(.+)$', item_text)
            if task_match:
                checked = 'checked' if task_match.group(1).lower() == 'x' else ''
                item_content = render_inline_formatting(task_match.group(2))
                html_lines.append(f'<li class="task-item"><input type="checkbox" disabled {checked}> {item_content}')
            else:
                html_lines.append(f'<li>{render_inline_formatting(item_text)}')
            list_stack[-1]['li_open'] = True
            i += 1
            continue

        # Continuation of an open list item
        if list_stack and list_stack[-1]['li_open'] and line[:1] in (' ', '\t'):
            html_lines.append("<br> " + render_inline_formatting(stripped))
            i += 1
            continue

        # Raw HTML island. CommonMark ends an HTML block at the first blank
        # line, which is what lets `<div align="center">` wrap markdown: the
        # tag is its own block and the prose inside is still parsed below.
        if looks_like_raw_html_block(stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            block_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip():
                block_lines.append(lines[j])
                j += 1
            i = j
            html_lines.append(sanitize_raw_html(
                '\n'.join(block_lines), src_rel_path, format_text=render_inline_formatting))
            continue

        # Standard Paragraph
        flush_table()
        flush_list()
        flush_blockquote()
        html_lines.append(f'<p>{render_inline_formatting(stripped)}</p>')
        i += 1

    flush_table()
    flush_list()
    flush_blockquote()

    full_html = "\n".join(html_lines)

    # STEP 4: Restore Protected Tokens
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    for idx, raw_code in enumerate(inline_codes):
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.lstrip("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)

        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'

        n_lines = len(lines_b) - 2
        collapsed = n_lines > 14
        wrapper_cls = 'code-wrapper collapsed' if collapsed else 'code-wrapper'
        expand_btn = (
            f'<button class="expand-btn" data-label="expand ▾ · {n_lines} lines" '
            f'onclick="toggleCode(this)">expand ▾ · {n_lines} lines</button>'
            if collapsed else ''
        )

        block_html = (
            f'<div class="{wrapper_cls}" data-lines="{n_lines}">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<span class="code-actions">{expand_btn}'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></span>'
            f'</div>'
            f'<pre><code{lang_attr}>{escaped_content}</code></pre>'
            f'</div>'
        )
        full_html = full_html.replace(f"___CODEBLOCK_{idx}___", block_html)

    return full_html, toc


def compute_rel_prefix(target_rel_path: str) -> str:
    """Calculate relative path back to root docs_html directory."""
    parts = Path(target_rel_path).parts
    if len(parts) <= 1:
        return "./"
    return "../" * (len(parts) - 1)


def build_sidebar_html(current_rel_path: str, rel_prefix: str) -> str:
    """Build the navigation sidebar HTML."""
    sidebar_sections: dict[str, list[str]] = {}

    for rel_path, part, display_title, _blurb in DOC_FILES:
        target_html_rel = rel_path.replace(".md", ".html")
        href = rel_prefix + target_html_rel
        is_active = (rel_path == current_rel_path) or (rel_path.replace(".md", ".html") == current_rel_path)
        active_cls = "active" if is_active else ""
        sidebar_sections.setdefault(part, []).append(
            f'<li class="nav-item"><a href="{href}" class="nav-link {active_cls}" title="{display_title}"><span class="nav-link-text">{display_title}</span></a></li>'
        )

    html_out = ['<div class="sidebar-search"><input type="text" id="navSearch" placeholder="Search docs..." onkeyup="filterNav()"></div>']

    for part, items in sidebar_sections.items():
        html_out.append('<div class="nav-group">')
        html_out.append(f'<div class="nav-group-title">{part}</div>')
        html_out.append(f'<ul class="nav-list">{"".join(items)}</ul>')
        html_out.append('</div>')

    return "\n".join(html_out)


def build_toc_html(toc_items: list[dict]) -> str:
    """Build the right sidebar table of contents."""
    if not toc_items:
        return '<div class="toc-empty">No section headings</div>'

    toc_links = []
    for item in toc_items:
        indent_cls = "toc-h3" if item['level'] == 3 else "toc-h2"
        toc_links.append(f'<li class="{indent_cls}"><a href="#{item["id"]}" class="toc-link">{item["title"]}</a></li>')

    return f'<ul class="toc-list">{"".join(toc_links)}</ul>'


def generate_html_page(rel_path: str, category: str, display_title: str):
    """Generate single HTML file for a markdown document."""
    src_file = WORKSPACE_DIR / rel_path
    if not src_file.exists():
        print(f"Warning: {src_file} does not exist, skipping.")
        return

    md_text = src_file.read_text(encoding="utf-8")

    word_count = len(md_text.split())
    reading_time = max(1, round(word_count / 200))

    html_body, toc_items = parse_markdown_to_html(md_text, rel_path)

    rel_prefix = compute_rel_prefix(rel_path)
    sidebar_html = build_sidebar_html(rel_path, rel_prefix)
    toc_html = build_toc_html(toc_items)

    current_idx = next((i for i, df in enumerate(DOC_FILES) if df[0] == rel_path), 0)
    prev_doc = DOC_FILES[current_idx - 1] if current_idx > 0 else None
    next_doc = DOC_FILES[current_idx + 1] if current_idx < len(DOC_FILES) - 1 else None

    prev_html = ""
    if prev_doc:
        prev_href = rel_prefix + prev_doc[0].replace(".md", ".html")
        prev_html = f'<a href="{prev_href}" class="nav-card prev-card"><span class="card-label">← Previous</span><span class="card-title">{prev_doc[2]}</span></a>'
    next_html = ""
    if next_doc:
        next_href = rel_prefix + next_doc[0].replace(".md", ".html")
        next_html = f'<a href="{next_href}" class="nav-card next-card"><span class="card-label">Next →</span><span class="card-title">{next_doc[2]}</span></a>'
    page_html = HEAD_TEMPLATE.format(
        rel_prefix=rel_prefix,
        title=f"{display_title} | Mamba-3-Lite Documentation",
        extra_head=DOC_EXTRA_HEAD,
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body>
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="{rel_prefix}index.html" class="brand-logo">
                <span class="brand-name">Mamba-3-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="{rel_prefix}index.html" class="header-link">Portal</a>
            <a href="{rel_prefix}README.html" class="header-link">README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Content Area -->
        <main class="main-content">
            <div class="content-container">
                <div class="breadcrumb">
                    <a href="{rel_prefix}index.html">Docs</a> &gt; <span>{category}</span> &gt; <span class="current">{display_title}</span>
                </div>

                <div class="doc-header">
                    <h1 class="doc-title">{display_title}</h1>
                    <div class="doc-meta">
                        <span class="meta-item"><span class="meta-mark">&sect;</span> {rel_path}</span>
                        <span class="meta-item"><span class="meta-mark">&para;</span> {word_count:,} words</span>
                        <span class="meta-item"><span class="meta-mark">&tau;</span> ~{reading_time} min read</span>
                    </div>
                </div>

                <article class="markdown-body" id="articleBody">
                    {html_body}
                </article>

                <div class="doc-footer-nav">
                    {prev_html}
                    {next_html}
                </div>
            </div>
        </main>

        <!-- Right Sidebar Table of Contents -->
        <aside class="toc-sidebar">
            <div class="toc-inner">
                <div class="toc-title">On This Page</div>
                {toc_html}
            </div>
        </aside>
    </div>

    <!-- Scripts -->
    <script defer src="{rel_prefix}assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / rel_path.replace(".md", ".html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page_html, encoding="utf-8")


def generate_index_portal():
    """Generate interactive index.html home portal."""
    sidebar_html = build_sidebar_html("index.html", "./")

    # The index is the book's table of contents. It is derived from DOC_FILES
    # so the sidebar and the index cannot disagree about the chapter list.
    parts: dict[str, list[str]] = {}
    chapter_no = 0
    for rel_path, part, display_title, blurb in DOC_FILES:
        chapter_no += 1
        parts.setdefault(part, []).append(f"""
            <a href="{rel_path.replace('.md', '.html')}" class="portal-card">
                <span class="card-tag">Ch {chapter_no}</span>
                <div class="card-body">
                    <h3 class="card-heading">{display_title}</h3>
                    <p class="card-desc">{blurb}</p>
                </div>
            </a>
            """)

    portal_cards_html = ""
    for part, cards in parts.items():
        n = len(cards)
        portal_cards_html += f"""
        <section class="portal-section">
            <header class="portal-section-head">
                <span class="portal-section-mark">{n} {'chapter' if n == 1 else 'chapters'}</span>
                <h2 class="portal-section-title">{part}</h2>
            </header>
            <div class="portal-grid">{''.join(cards)}</div>
        </section>
        """

    # Hero and pass-diagram animations live in assets/portal.js so the
    # index and every doc page share one script surface.
    index_scripts = '<script defer src="assets/portal.js"></script>'

    index_html = HEAD_TEMPLATE.format(
        title="Mamba-3-Lite Documentation Portal",
        extra_head="",
        rel_prefix="./",
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body class="index-portal">
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="index.html" class="brand-logo">
                <span class="brand-name">Mamba-3-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="README.html" class="header-link">GitHub README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Portal Content -->
        <main class="main-content">
            <div class="content-container">
                <div class="hero-banner">
                    <h1 class="hero-title">Mamba-3-Lite</h1>
                    <p class="hero-subtitle">From-scratch PyTorch reproduction of Mamba-3 at ~434M params — complex-valued SSD state spaces (N=64, complex64), a fully-connected MIMO head mixer, and zero causal convolution. Pure PyTorch, one sanctioned opt-in Triton kernel. Each document shows the arithmetic, not just the result.</p>

                    <div class="mamba-telemetry-ribbon" aria-label="Key Mamba-3 Architectural Metrics">
                        <div class="telemetry-card terra">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> STATE COMPRESSION</div>
                            <div class="tc-val">50% <span class="unit">SMALLER N</span></div>
                            <div class="tc-desc">N=64 complex64 packs two real sub-states into one complex state (vs Mamba-2 N=128)</div>
                        </div>
                        <div class="telemetry-card olive">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> MIMO MIXER</div>
                            <div class="tc-val">16&times;16 <span class="unit">CROSS-HEAD</span></div>
                            <div class="tc-desc">Fully-connected mixer replaces SISO constraint &middot; zero sequence cost overhead</div>
                        </div>
                        <div class="telemetry-card gold">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> CAUSAL CONV</div>
                            <div class="tc-val">0 <span class="unit">ELIMINATED</span></div>
                            <div class="tc-desc">Zero causal_conv1d passes &middot; pure chunked linear projection saves memory bandwidth</div>
                        </div>
                        <div class="telemetry-card ink">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> COMPLEX EIGEN</div>
                            <div class="tc-val">&alpha;+i&omega; <span class="unit">DECAY+ROT</span></div>
                            <div class="tc-desc">Complex exponential captures both decay (&alpha;) and oscillation (&omega;) per state</div>
                        </div>
                    </div>

                    <div class="hero-figure" id="heroFigure" role="region" aria-label="Figure A0: Complex-Valued State-Space Dynamics and 32 Conjugate Phasor Orbits">
                        <div class="hero-figure-header">
                            <div class="fig-badge">
                                <span class="fig-dot" aria-hidden="true"></span>
                                <span class="fig-tag">FIG. A0</span>
                                <span class="fig-sep" aria-hidden="true">/</span>
                                <span class="fig-title">COMPLEX STATE-SPACE DYNAMICS</span>
                                <span class="fig-dim">N=64 &middot; 32 CONJUGATE PAIRS</span>
                            </div>
                            <div class="fig-controls">
                                <button class="fig-btn active" data-mode="spiral" type="button" title="Logarithmic Spiral Manifolds &amp; Phasors">SPIRAL &middot; PHASORS</button>
                                <button class="fig-btn" data-mode="orbits" type="button" title="Conjugate Phase Orbits on Unit Disk">ORBIT TRACKS</button>
                                <button class="fig-btn" data-mode="field" type="button" title="Continuous Decay Flow Field">DECAY FIELD</button>
                                <button class="fig-btn fig-btn-icon" id="figPauseBtn" type="button" title="Pause / Resume Animation" aria-label="Pause animation">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="fig-canvas-wrap">
                            <canvas id="complexSpiralCanvas" class="hero-canvas" aria-hidden="true"></canvas>
                            <div class="fig-overlay-hud" aria-hidden="true">
                                <div class="hud-corner top-left">
                                    <span class="hud-lbl">EIGEN-SPECTRUM</span>
                                    <span class="hud-val">&lambda;_k = &alpha;_k &plusmn; i&omega;_k</span>
                                </div>
                                <div class="hud-corner top-right">
                                    <span class="hud-lbl">PHASE VELOCITY</span>
                                    <span class="hud-val">&omega; &in; [0.35, 2.20] rad/dt (illustrative)</span>
                                </div>
                                <div class="hud-corner bottom-left">
                                    <span class="hud-lbl">STATE NORM</span>
                                    <span class="hud-val">||h_t||_2 = <span class="num" id="hudNormVal">1.414</span></span>
                                </div>
                                <div class="hud-corner bottom-right">
                                    <span class="hud-lbl">DISCRETIZATION</span>
                                    <span class="hud-val">ZOH &middot; &Delta;t = softplus(dt) (per-token, illustrative)</span>
                                </div>
                                <div class="hud-probe" id="hudProbe" style="opacity: 0;">
                                    <div class="probe-card">
                                        <span class="probe-tag" id="probeTag">PAIR #16</span>
                                        <span class="probe-val" id="probeCoords">Re: +0.642 &middot; Im: &plusmn;0.518</span>
                                        <span class="probe-sub" id="probeDecay">Decay &tau; = 18.4 steps &middot; &theta; = 38.8&deg;</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="hero-figure-footer">
                            <div class="fig-formula">
                                <span class="formula-sym">h<sub class="f-sub">t</sub></span>
                                <span class="formula-op">=</span>
                                <span class="formula-term">exp((<span class="term-a">A</span> + i<span class="term-b">B</span>)&Delta;t)</span>
                                <span class="formula-dot">&middot;</span>
                                <span class="formula-sym">h<sub class="f-sub">t-1</sub></span>
                                <span class="formula-op">+</span>
                                <span class="formula-term">B<sub class="f-sub">t</sub>x<sub class="f-sub">t</sub></span>
                            </div>
                            <div class="fig-legend">
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">Im+ (&omega; &gt; 0, CCW)</span></span>
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">Im- (&omega; &lt; 0, CW)</span></span>
                                <span class="legend-item"><span class="legend-swatch core"></span> <span class="legend-label">State Core h<sub class="f-sub">t</sub></span></span>
                                <span class="legend-hint">CLICK: INJECT TOKEN &middot; HOVER: PROBE PAIR</span>
                            </div>
                        </div>
                    </div>
                    <div class="spec-sheet">
                        <div class="spec-sheet-rule" aria-hidden="true">
                            <span class="spec-rule-key">DATASHEET</span>
                            <span class="spec-rule-meta">rev 0.1 &middot; chk bf16 &middot; 8.0B tok / A100-80</span>
                        </div>
                        <dl class="spec-grid">
                            <div class="spec-cell"><dt class="spec-key">01 &middot; params</dt><dd class="spec-val">~434<span class="unit">M</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">02 &middot; layers</dt><dd class="spec-val">28</dd></div>
                            <div class="spec-cell"><dt class="spec-key">03 &middot; state</dt><dd class="spec-val">N=64<span class="unit">  complex64</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">04 &middot; hidden</dt><dd class="spec-val">1024</dd></div>
                            <div class="spec-cell"><dt class="spec-key">05 &middot; ffn</dt><dd class="spec-val">2048<span class="unit">  SwiGLU</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">06 &middot; vocab</dt><dd class="spec-val">50 K</dd></div>
                            <div class="spec-cell"><dt class="spec-key">07 &middot; ctx</dt><dd class="spec-val">2048<span class="unit">  chunk 64</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">08 &middot; tokens</dt><dd class="spec-val">8.0<span class="unit">  B</span></dd></div>
                        </dl>
                    </div>

                    <div class="mechanism-section" aria-label="Interactive Mamba-3 Core Mechanisms">
                        <div class="mechanism-section-head">
                            <span class="mechanism-section-title">&sect; MAMBA-3 CORE MECHANISMS &middot; INTERACTIVE BENCHMARK LABS</span>
                        </div>
                        <div class="mechanism-grid">
                            <!-- Card 1: Complex SSD State Explorer -->
                            <div class="mechanism-card card-ssd" id="mchCardSsd">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">01 &middot; COMPLEX SSD</span>
                                    <span class="mch-title">Eigenvalue Spectrum Explorer</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Visualize how complex eigenvalues &lambda; = &alpha; + i&omega; govern state dynamics. Decay rate &alpha; controls memory retention; rotation &omega; enables oscillatory patterns impossible in real SSMs.</p>
                                    <div class="mch-control-row">
                                        <span>Decay Range (&alpha;):</span>
                                        <strong id="ssdDecayLabel">[-3.0, -0.05]</strong>
                                    </div>
                                    <input type="range" id="ssdDecaySlider" class="mch-slider" min="1" max="50" step="1" value="15" aria-label="Decay magnitude">
                                    <canvas id="ssdEigenCanvas" class="mch-canvas" width="280" height="160" aria-hidden="true"></canvas>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Active Eigen Pairs:</span>
                                            <span id="ssdPairCount">32 conjugate pairs</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>Memory Retention (midpoint &alpha; half-life):</span>
                                            <span id="ssdHalfLife">~0.4 steps</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>State Capacity vs Real SSM:</span>
                                            <span id="ssdCapacity">2.0&times; per parameter</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="ssdInjectBtn">&#9654; Inject Token &amp; Observe Decay</button>
                                </div>
                            </div>

                            <!-- Card 2: MIMO Head Mixer Matrix -->
                            <div class="mechanism-card card-mimo" id="mchCardMimo">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">02 &middot; MIMO MIXER</span>
                                    <span class="mch-title">Cross-Head Information Flow</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Simulate the 16&times;16 fully-connected mixer across SSM heads. Identity-initialized weights gradually learn cross-head correlations during training, replacing SISO isolation.</p>
                                    <div class="mimo-matrix-grid" id="mimoMatrixGrid">
                                        <!-- 16x16 mini cells generated via JS -->
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Mixer Dimensions:</span>
                                            <span>1024 &rarr; 1024 (16&times;64)</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>Init Strategy:</span>
                                            <span>Pure identity (eye_) &middot; init_std skipped</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>Active Cross-Head Links:</span>
                                            <span id="mimoActiveLinks">0 of 240 off-diagonal</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="mimoStepBtn">&#9654; Forward Pass &amp; Update Weights</button>
                                </div>
                            </div>

                            <!-- Card 3: Chunkwise Recurrence Pipeline -->
                            <div class="mechanism-card card-chunk" id="mchCardChunk">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">03 &middot; CHUNKWISE SSD</span>
                                    <span class="mch-title">Linear Scan vs Chunked Parity</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Compare naive O(T) sequential scan against the chunkwise algorithm (Q=64). The chunkwise approach decomposes into parallelizable intra-chunk and inter-chunk passes.</p>
                                    <div class="chunk-pipeline-box" id="chunkPipelineDisplay">
                                        <div class="chunk-stage">
                                            <span class="chunk-badge scan">NAIVE</span>
                                            <span>s_t = A&#772;&#183;s_&#123;t-1&#125; + B_t&#183;x_t, A&#772;=exp(softplus(dt)&#183;A) (O(T) loop)</span>
                                        </div>
                                        <div class="chunk-stage">
                                            <span class="chunk-badge chunk">CHUNK</span>
                                            <span>L[l,s]=exp(A_cs[l]-A_cs[s])&#183;1[l&ge;s]; Y_diag = einsum(C,B,L,X) (intra-chunk matmul)</span>
                                        </div>
                                        <div class="chunk-stage">
                                            <span class="chunk-badge state">STATE</span>
                                            <span>s_chunk = einsum(B,decay,X); cross-chunk via decay_chunk matrix (exp of chunk-decay differences)</span>
                                        </div>
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Chunk Size (Q):</span>
                                            <span id="chunkSizeVal">64 tokens</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>Sequence Length:</span>
                                            <span>2,048 (32 chunks)</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>Numerical Match (vs naive):</span>
                                            <span id="chunkMatchVal">&lt; 1e-5 max |&Delta;|</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="chunkVerifyBtn">&#9654; Verify Chunkwise &equiv; Naive</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="pass-widget" id="passWidget" role="region" aria-label="Figure A1: Complete Training Step Pipeline — Forward Activations, Autograd Backward Pass, Gradient Checkpointing Re-computation, and AdamW Step">
                        <div class="pass-widget-header">
                            <div class="pass-badge">
                                <span class="pass-dot" aria-hidden="true"></span>
                                <span class="pass-tag">FIG. A1</span>
                                <span class="pass-sep" aria-hidden="true">/</span>
                                <span class="pass-title">FULL TRAINING STEP PIPELINE</span>
                                <span class="pass-dim">FORWARD &middot; AUTOGRAD BACKWARD &middot; ADAMW</span>
                            </div>
                            <div class="pass-controls">
                                <button class="pass-btn active" data-phase="cycle" type="button" title="Full Forward / Backward / AdamW Cycle">AUTO CYCLE</button>
                                <button class="pass-btn" data-phase="forward" type="button" title="Inspect Forward Activation Flow">FORWARD</button>
                                <button class="pass-btn" data-phase="backward" type="button" title="Inspect Backward Gradient Flow">BACKWARD</button>
                                <button class="pass-btn pass-btn-icon" id="passPauseBtn" type="button" title="Pause / Resume Pipeline" aria-label="Pause pipeline">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="pass-canvas-wrap">
                            <canvas id="passDiagramCanvas" class="pass-canvas" aria-hidden="true"></canvas>
                            <div class="pass-overlay-hud" aria-hidden="true">
                                <div class="pass-hud-chip top-left">
                                    <span class="ph-lbl">CURRENT PHASE</span>
                                    <span class="ph-val" id="phCurrentPhase">FORWARD ACTIVATIONS</span>
                                </div>
                                <div class="pass-hud-chip top-right">
                                    <span class="ph-lbl">MEMORY CONTRACT</span>
                                    <span class="ph-val" id="phStepMetrics">grad_checkpoint: TRUE &middot; re-compute activations in backward</span>
                                </div>
                                <div class="pass-stage-tooltip" id="passStageTooltip" style="opacity: 0;">
                                    <div class="st-card">
                                        <span class="st-tag" id="stTag">STAGE 03: COMPLEX SSD SCAN</span>
                                        <span class="st-op" id="stOp">s_t = exp(softplus(dt)&#183;A)&#183;s_{{t-1}} + B_t x_t (A complex64)</span>
                                        <span class="st-shape" id="stShape">Output Y: [B, 2048, 16, 64] bf16</span>
                                        <span class="st-desc" id="stDesc">Chunkwise linear recurrence (Q=64) — pure-PyTorch einsum by default, opt-in Triton kernel</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pass-widget-footer">
                            <div class="pass-status-ticker">
                                <span class="ticker-beacon" id="passTickerBeacon">&bull;</span>
                                <span class="ticker-text" id="passTickerText">FORWARD &middot; Tokens x_t &rarr; Fused Embed &rarr; 28&times; Complex SSD Blocks &rarr; Full-Vocab CE Loss</span>
                            </div>
                            <div class="pass-legend">
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">Forward Activations</span></span>
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">Backward Gradients (&part;&ell;/&part;&theta;)</span></span>
                                <span class="legend-item"><span class="legend-swatch gold"></span> <span class="legend-label">AdamW Update (&Delta;&theta;)</span></span>
                                <span class="legend-hint">CLICK STAGE: INSPECT &middot; HOVER: TENSOR SHAPES</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="portal-content">
                    {portal_cards_html}
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    {index_scripts}
</body>
</html>
"""

    out_file = OUTPUT_DIR / "index.html"
    out_file.write_text(index_html, encoding="utf-8")


def generate_assets():
    """Copy assets/style.css and assets/portal.js into the docs build."""
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    src_css = WORKSPACE_DIR / "assets" / "style.css"
    shutil.copyfile(src_css, assets_dir / "style.css")
    src_js = WORKSPACE_DIR / "assets" / "portal.js"
    if src_js.exists():
        shutil.copyfile(src_js, assets_dir / "portal.js")


def main():
    print("Building Mamba-3-Lite HTML Documentation...")
    # docs_html/ is a build artifact, so wipe it first. Otherwise a chapter
    # dropped from DOC_FILES survives as a stale page and gets published.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".nojekyll").touch()

    generate_assets()

    for rel_path, part, display_title, _blurb in DOC_FILES:
        print(f"Generating: {rel_path} -> docs_html/{rel_path.replace('.md', '.html')}")
        generate_html_page(rel_path, part, display_title)

    generate_index_portal()
    print("\nDocumentation build complete!")
    print(f"HTML Portal location: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
