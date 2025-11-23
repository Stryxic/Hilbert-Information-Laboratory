"""
vault_manager.py
Hilbert Information Laboratory - Vault subsystem

A simple Obsidian-style vault implementation for managing notes,
metadata, backlinks, and graph extraction for the Hilbert pipeline.

A vault is a directory with the structure:

vault/
    vault.json
    notes/
        <uuid>.md
    attachments/
        ...
    
Each note is a Markdown file with optional YAML frontmatter:

---
id: <uuid>
title: "Some title"
tags: ["hilbert", "lsa"]
created: 2025-11-21T12:00
modified: 2025-11-21T12:00
---

Markdown content...

This module provides functions to:
- create/open vaults
- create/read/update/delete notes
- list/search notes
- extract wikilinks and backlinks
- build graph structures compatible with the Hilbert pipeline
"""

from __future__ import annotations
import os
import json
import uuid
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Utility: YAML frontmatter
# -----------------------------

FRONTMATTER_RE = re.compile(r"^---\s*(.*?)---\s*(.*)$", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

def parse_frontmatter(text: str) -> Tuple[Dict, str]:
    """
    Extract YAML frontmatter from a markdown file.
    Returns: (metadata_dict, content_string)
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    yaml_block, content = m.groups()
    meta = {}

    for line in yaml_block.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # simple scalar parsing
            if val.startswith("[") and val.endswith("]"):
                # parse tag lists
                val = [x.strip().strip('"').strip("'") for x in val[1:-1].split(",") if x.strip()]
            meta[key] = val

    return meta, content


def render_frontmatter(meta: Dict, content: str) -> str:
    """Reconstruct YAML frontmatter and content."""
    fm_lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, list):
            v = "[" + ", ".join(f'"{x}"' for x in v) + "]"
        fm_lines.append(f"{k}: {v}")
    fm_lines.append("---\n")
    return "\n".join(fm_lines) + content


# -----------------------------
# Vault Manager Class
# -----------------------------

class VaultManager:
    """
    Filesystem-based vault manager.
    """

    def __init__(self, vault_path: str):
        self.vault_path = os.path.abspath(vault_path)
        self.notes_path = os.path.join(self.vault_path, "notes")
        self.attach_path = os.path.join(self.vault_path, "attachments")
        self.meta_file = os.path.join(self.vault_path, "vault.json")

        if not os.path.isdir(self.vault_path):
            raise FileNotFoundError(f"Vault does not exist: {self.vault_path}")

        if not os.path.isdir(self.notes_path):
            os.makedirs(self.notes_path, exist_ok=True)

        if not os.path.isdir(self.attach_path):
            os.makedirs(self.attach_path, exist_ok=True)

        if not os.path.exists(self.meta_file):
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump({"vault_created": datetime.utcnow().isoformat()}, f, indent=2)

    # ---------------------------------------------------------
    # Vault operations
    # ---------------------------------------------------------

    @staticmethod
    def create_vault(path: str) -> "VaultManager":
        """Create a new vault directory structure."""
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "notes"), exist_ok=True)
        os.makedirs(os.path.join(path, "attachments"), exist_ok=True)

        meta = {
            "vault_created": datetime.utcnow().isoformat(),
            "version": "1.0"
        }

        with open(os.path.join(path, "vault.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return VaultManager(path)

    # ---------------------------------------------------------
    # Note CRUD
    # ---------------------------------------------------------

    def list_notes(self) -> List[str]:
        """Return a list of note IDs."""
        ids = []
        for fname in os.listdir(self.notes_path):
            if fname.endswith(".md"):
                ids.append(fname[:-3])
        return ids

    def get_note_path(self, note_id: str) -> str:
        return os.path.join(self.notes_path, f"{note_id}.md")

    def load_note(self, note_id: str) -> Dict:
        """Load note metadata + content."""
        path = self.get_note_path(note_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such note {note_id}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        meta, content = parse_frontmatter(text)
        meta.setdefault("id", note_id)
        return {"meta": meta, "content": content}

    def save_note(self, note_id: str, meta: Dict, content: str) -> None:
        """Write note back to disk."""
        meta["modified"] = datetime.utcnow().isoformat()

        text = render_frontmatter(meta, content)
        path = self.get_note_path(note_id)

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def create_note(self, title: str = "Untitled") -> str:
        """Create a new note with generated UUID."""
        note_id = uuid.uuid4().hex[:12]

        meta = {
            "id": note_id,
            "title": title,
            "tags": [],
            "created": datetime.utcnow().isoformat(),
            "modified": datetime.utcnow().isoformat(),
        }

        content = f"# {title}\n\n"
        self.save_note(note_id, meta, content)
        return note_id

    def delete_note(self, note_id: str) -> None:
        path = self.get_note_path(note_id)
        if os.path.exists(path):
            os.remove(path)

    # ---------------------------------------------------------
    # Search, links, backlinks
    # ---------------------------------------------------------

    def search(self, query: str) -> List[str]:
        """Simple substring search across notes."""
        matches = []
        for nid in self.list_notes():
            data = self.load_note(nid)
            if query.lower() in data["content"].lower() or query.lower() in json.dumps(data["meta"]).lower():
                matches.append(nid)
        return matches

    def extract_links(self, note_id: str) -> List[str]:
        """Extract wikilinks [[note-id or title]]."""
        data = self.load_note(note_id)
        return WIKILINK_RE.findall(data["content"])

    def build_backlinks(self) -> Dict[str, List[str]]:
        """
        Return a mapping: note_id -> list of notes that link to it.
        """
        backlinks = {nid: [] for nid in self.list_notes()}

        for src in self.list_notes():
            links = self.extract_links(src)
            for link in links:
                for target in self.list_notes():
                    title = self.load_note(target)["meta"].get("title", "")
                    if link == target or link == title:
                        backlinks[target].append(src)

        return backlinks

    # ---------------------------------------------------------
    # Graph extraction for UI
    # ---------------------------------------------------------

    def build_graph(self) -> Dict:
        """
        Build a graph representation suitable for the front-end graph viewer.
        Nodes: notes
        Edges: wikilinks + backlinks
        """
        nodes = []
        edges = []

        note_ids = self.list_notes()
        backlinks = self.build_backlinks()

        for nid in note_ids:
            info = self.load_note(nid)
            nodes.append({
                "id": f"note:{nid}",
                "type": "note",
                "label": info["meta"].get("title", nid)
            })

        for src in note_ids:
            links = self.extract_links(src)
            for link in links:
                for target in note_ids:
                    title = self.load_note(target)["meta"].get("title", "")
                    if link == target or link == title:
                        edges.append({
                            "source": f"note:{src}",
                            "target": f"note:{target}",
                            "relation": "link"
                        })

        return {"nodes": nodes, "edges": edges}
