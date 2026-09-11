"""tree-sitter tag extraction for the repo map.

AST is a per-file, use-once intermediate: parse -> tag queries -> (name,
line) tags -> discard the tree; the graph/ranking layers only see tags.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

try:  # optional dependency: pip install gencode[map]
    from tree_sitter_language_pack import get_language, get_parser
    _PACK_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on environment
    get_parser = None
    get_language = None
    _PACK_IMPORT_ERROR = str(exc)

# Offline fallback: single-language wheels expose a plain `language()` capsule.
_FALLBACK_LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "tsx": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "rust": "tree_sitter_rust",
}

# language name -> (parser, ts_language) once resolved; never guessed twice.
_resolved_languages = {}
_unresolvable_languages = set()
_broken_queries = set()
_pack_broken = False  # pack download failed once -> stop retrying


@dataclass(frozen=True)
class FileTags:
    path: str
    language: str
    sha256: str
    definitions: tuple  # ((name, line), ...)
    references: tuple  # ((name, line), ...)


# Known suffixes only; unknown file types are skipped, never guessed.
SUFFIX_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
}

_TS_DEFINITION_QUERY = """
    (function_declaration name: (identifier) @name.definition.function)
    (class_declaration name: (identifier) @name.definition.class)
    (method_definition name: (property_identifier) @name.definition.method)
    (interface_declaration name: (type_identifier) @name.definition.class)
    (type_alias_declaration name: (type_identifier) @name.definition.class)
"""
_TS_REFERENCE_QUERY = "[(identifier) (property_identifier) (type_identifier) (nested_type_identifier)] @name.reference"

# Definition captures per language family. A query that fails to compile is
# dropped once at load time; the language then yields no definitions.
_DEFINITION_QUERIES = {
    "python": """
        (function_definition name: (identifier) @name.definition.function)
        (class_definition name: (identifier) @name.definition.class)
    """,
    "javascript": """
        (function_declaration name: (identifier) @name.definition.function)
        (class_declaration name: (identifier) @name.definition.class)
        (method_definition name: (property_identifier) @name.definition.method)
        (lexical_declaration (variable_declarator name: (identifier) @name.definition.function))
    """,
    "typescript": _TS_DEFINITION_QUERY,
    "tsx": _TS_DEFINITION_QUERY,
    "go": """
        (function_declaration name: (identifier) @name.definition.function)
        (method_declaration name: (field_identifier) @name.definition.method)
        (type_spec name: (type_identifier) @name.definition.class)
    """,
    "java": """
        (method_declaration name: (identifier) @name.definition.method)
        (class_declaration name: (identifier) @name.definition.class)
        (interface_declaration name: (identifier) @name.definition.class)
        (enum_declaration name: (identifier) @name.definition.class)
    """,
    "c": """
        (function_definition declarator: (function_declarator declarator: (identifier) @name.definition.function))
        (struct_specifier name: (type_identifier) @name.definition.class)
        (type_definition declarator: (type_identifier) @name.definition.class)
    """,
    "cpp": """
        (function_definition declarator: (function_declarator declarator: (identifier) @name.definition.function))
        (function_definition declarator: (function_declarator declarator: (field_identifier) @name.definition.method))
        (struct_specifier name: (type_identifier) @name.definition.class)
        (class_specifier name: (type_identifier) @name.definition.class)
    """,
    "rust": """
        (function_item name: (identifier) @name.definition.function)
        (struct_item name: (type_identifier) @name.definition.class)
        (enum_item name: (type_identifier) @name.definition.class)
        (impl_item type: (type_identifier) @name.definition.class)
        (trait_item name: (identifier) @name.definition.class)
    """,
}

# Reference captures: identifier-like leaf nodes, including member/type
# positions, so cross-file name matching stays cheap and language-agnostic.
_REFERENCE_QUERIES = {
    "python": "[(identifier) (attribute attribute: (identifier))] @name.reference",
    "javascript": "[(identifier) (property_identifier) (shorthand_property_identifier)] @name.reference",
    "typescript": _TS_REFERENCE_QUERY,
    "tsx": _TS_REFERENCE_QUERY,
    "go": "[(identifier) (field_identifier) (type_identifier) (package_identifier)] @name.reference",
    "java": "[(identifier) (field_access field: (identifier))] @name.reference",
    "c": "[(identifier) (field_identifier) (type_identifier)] @name.reference",
    "cpp": "[(identifier) (field_identifier) (type_identifier) (namespace_identifier)] @name.reference",
    "rust": "[(identifier) (field_identifier) (type_identifier) (scoped_identifier)] @name.reference",
}

_compiled_queries = {}


def _resolve_fallback(language):
    """Resolve via single-language wheels when the language pack is missing
    or its grammar download is unavailable (offline environments)."""
    try:
        from tree_sitter import Language, Parser

        module_name = _FALLBACK_LANGUAGE_MODULES.get(language)
        if not module_name:
            return None
        import importlib

        module = importlib.import_module(module_name)
        capsule_fn = getattr(module, f"language_{language}", None) or module.language
        ts_language = Language(capsule_fn())
        parser = Parser(ts_language)
        return parser, ts_language
    except Exception:
        return None


def resolve_language(language):
    """Return (parser, ts_language) or None; wheel first, pack second."""
    global _pack_broken
    if language in _resolved_languages:
        return _resolved_languages[language]
    if language in _unresolvable_languages:
        return None
    resolved = _resolve_fallback(language)
    if resolved is None and get_parser is not None and get_language is not None and not _pack_broken:
        try:
            resolved = (get_parser(language), get_language(language))
        except Exception:
            _pack_broken = True
            resolved = None
    if resolved is None:
        _unresolvable_languages.add(language)
    else:
        _resolved_languages[language] = resolved
    return resolved


_pack_available_cache = None


def pack_available():
    """True when at least one supported language can actually be parsed."""
    global _pack_available_cache
    if _pack_available_cache is None:
        _pack_available_cache = (
            bool(_resolved_languages)
            or _resolve_fallback("python") is not None
            or get_parser is not None
        )
    return _pack_available_cache


def availability_reason():
    if pack_available():
        return "ok"
    reason = f"tree_sitter_language_pack_missing: {_PACK_IMPORT_ERROR}" if _PACK_IMPORT_ERROR else "no_tree_sitter_backend"
    return reason


def language_for_path(path):
    return SUFFIX_LANGUAGES.get(Path(path).suffix.lower())


def _query(language, source):
    """Compile a query into a captures(root) callable, across API versions."""
    key = (language, source)
    if key in _compiled_queries:
        return _compiled_queries[key]
    if key in _broken_queries:
        return None
    captures_fn = None
    resolved = resolve_language(language)
    if resolved is not None:
        ts_language = resolved[1]
        try:  # py-tree-sitter >= 0.23
            from tree_sitter import Query, QueryCursor

            captures_fn = QueryCursor(Query(ts_language, source)).captures
        except Exception:
            captures_fn = None
        if captures_fn is None:
            try:  # py-tree-sitter < 0.23
                captures_fn = ts_language.query(source).captures
            except Exception:
                captures_fn = None
    if captures_fn is None:
        _broken_queries.add(key)
    _compiled_queries[key] = captures_fn
    return captures_fn


def _captures(captures_fn, root_node):
    """Yield (capture_name, node) across return-shape variants."""
    try:
        captured = captures_fn(root_node)
    except Exception:
        return
    if isinstance(captured, dict):
        for name, nodes in captured.items():
            for node in nodes:
                yield str(name), node
    else:
        for item in captured:
            if isinstance(item, tuple) and len(item) == 2:
                yield str(item[1]), item[0]


def _node_line(node):
    return int(node.start_point[0]) + 1


def extract_file_tags(abs_path, rel_path, language, source_bytes):
    """Parse one file and collect definition/reference tags, or None."""
    if language is None:
        return None
    resolved = resolve_language(language)
    if resolved is None:
        from .fallback import extract_fallback_tags

        definitions, references = extract_fallback_tags(rel_path, language, source_bytes)
        return FileTags(
            path=rel_path,
            language=language,
            sha256=hashlib.sha256(source_bytes).hexdigest(),
            definitions=definitions,
            references=references,
        )
    parser = resolved[0]
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    def_query = _query(language, _DEFINITION_QUERIES[language])
    ref_query = _query(language, _REFERENCE_QUERIES[language])

    definitions = []
    def_name_starts = set()
    if def_query is not None:
        for name, node in _captures(def_query, root_node):
            if not str(name).startswith("name.definition"):
                continue
            text = source_bytes[node.start_byte : node.end_byte]
            if text:
                definitions.append((text.decode("utf-8", "replace"), _node_line(node)))
                def_name_starts.add(node.start_byte)

    references = []
    seen_starts = set()
    if ref_query is not None:
        for name, node in _captures(ref_query, root_node):
            if not str(name).startswith("name.reference"):
                continue
            if node.start_byte in def_name_starts or node.start_byte in seen_starts:
                continue
            text = source_bytes[node.start_byte : node.end_byte]
            if text:
                references.append((text.decode("utf-8", "replace"), _node_line(node)))
                seen_starts.add(node.start_byte)

    return FileTags(
        path=rel_path,
        language=language,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        definitions=tuple(sorted(set(definitions), key=lambda item: (item[1], item[0]))),
        references=tuple(sorted(set(references), key=lambda item: (item[1], item[0]))),
    )
