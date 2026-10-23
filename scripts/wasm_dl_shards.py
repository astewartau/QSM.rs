#!/usr/bin/env python3
"""Which deep-learning wasm32 shards of unit-tests.yml a change needs.

unit-tests.yml runs tests/wasm_pipeline.rs on wasm32 four times: once in the plain build (the
`classical` shard, which every Rust change runs) and three times in the deep-learning build
(`onnx simd`), split by test name into the shards `stages`, `masking` and `segmentation`. This
script decides which of those three a set of changed files needs.

How a file is classified, first match wins:

1. ALWAYS: build inputs, the wasm test itself, the weights fetcher, the CI that runs it, and this
   script. Every shard.
2. NEUTRAL: module roots that only declare modules and re-export items (checked: no fn, struct,
   impl, const, ... outside `#[cfg(test)]`). No shard, as in integration-tests.yml's path map.
   Registering a new classical algorithm touches them, and that must not run the networks.
3. Deep-learning code, per shard: the ROOTS (the network wrappers and model plumbing, plus what the
   test's phantom is built with), the WIRING (pipeline dispatchers that hold a network's entry
   point next to classical ones), and every file either of them reaches. Reach is the module
   graph of `crate::`/`super::`/`self::` paths and `use` trees in src/, read from the source on
   every run, so a helper a network starts calling joins its shard without anyone listing it. A
   ROOTS file contributes all of its paths; a WIRING file only those in its
   `#[cfg(feature = "onnx")]` items. The rest of a dispatcher (input checks, the classical arms,
   config defaults) runs the same in both builds, and the classical shard runs it.
4. CLASSICAL: src files no deep-learning shard reaches. No shard: the classical shard runs the
   same stage tests on them in the plain build. What the deep-learning build adds is the
   networks, the code they call, and the `onnx`/`simd` feature code, all of which is in 3.
5. Anything else under src/ (a new file, most likely) is unclassified and runs every shard.
   Files outside src/ that are not ALWAYS cannot reach the wasm build and run nothing.

`check` keeps the lists honest, and fails CI when they are not: every listed file exists, no
CLASSICAL file is reachable from a network, every NEUTRAL file is declarations only, and every
file that uses the model machinery (`crate::models`, `tract`, `feature = "onnx"` or
`feature = "simd"`) is a ROOTS or WIRING file or reached from one. The last one is what stops a
new network wrapper that nobody listed from skipping the shards: listed as CLASSICAL or left out
of ROOTS, it fails here.

Usage:
  wasm_dl_shards.py select FILE...       selection for these changed paths (a dry run)
  wasm_dl_shards.py select --from LIST   the same, paths read one per line from LIST
  wasm_dl_shards.py check                the list check above
  wasm_dl_shards.py graph FILE           which shards reach FILE, and through what

`select` prints its reasoning, then one JSON list of shards on the last line. With
`--github-output`, it also writes `wasm_dl_shards=<list>` to $GITHUB_OUTPUT.
"""

import argparse
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARDS = ("stages", "masking", "segmentation")

ALWAYS = [
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/*",
    "build.rs",
    "rust-toolchain",
    "rust-toolchain.toml",
    "tests/wasm_pipeline.rs",
    "tests/support/*",
    "scripts/fetch_model_weights.py",
    "scripts/wasm_dl_shards.py",
    ".github/workflows/unit-tests.yml",
    ".github/workflows/integration-tests.yml",
]

NEUTRAL = [
    "src/lib.rs",
    "src/bet/mod.rs",
    "src/bgremove/mod.rs",
    "src/inversion/mod.rs",
    "src/kernels/mod.rs",
    "src/pipeline/mod.rs",
    "src/solvers/mod.rs",
    "src/utils/mod.rs",
]

# Every shard builds the phantom (tests/wasm_pipeline.rs: `Phantom::new` uses the dipole kernel
# and the FFT on a `Grid`), resolves weights through the registry and runs tract.
_EVERY = [
    "src/models/mod.rs",
    "src/models/registry.rs",
    "src/models/onnx.rs",
    "src/models/download.rs",
    "src/kernels/dipole.rs",
    "src/fft.rs",
    "src/grid.rs",
]

ROOTS = {
    "stages": _EVERY + [
        # dl_bfr_*: BFRnet through run_bg_removal, iQFM through run_iqfm.
        "src/bgremove/bfrnet.rs",
        "src/inversion/iqfm.rs",
        # dl_inv_*: the inversion networks, and the overlap-tiling they share.
        "src/inversion/xqsm.rs",
        "src/inversion/qsmnet.rs",
        "src/inversion/qsmgan.rs",
        "src/inversion/lpcnn.rs",
        "src/inversion/ir2qsm.rs",
        "src/inversion/modl_qsm.rs",
        "src/inversion/autoqsm.rs",
        "src/inversion/nextqsm.rs",
        "src/inversion/iqsm.rs",
        "src/inversion/iqsm_plus.rs",
        "src/inversion/tiled.rs",
        "src/inversion/tile_budget.rs",
        # dl_sep_*: R2PRIMEnet, SUSEP-Net, χ-sepnet.
        "src/utils/r2primenet.rs",
        "src/separation/susep_net.rs",
        "src/separation/chisepnet.rs",
        # The `simd` feature is only on in the deep-learning build, and this shard also runs the
        # classical stage tests in it, so it is the only place the SIMD paths run on wasm32.
        "src/utils/simd_ops.rs",
    ],
    # dl_mask_*: HD-BET, RS2-Net, MindGrab through run_masking.
    "masking": _EVERY + [
        "src/bet/hdbet.rs",
        "src/bet/rs2net.rs",
        "src/bet/mindgrab.rs",
    ],
    # dl_seg_*: SynthSeg, whole volume and in windows.
    "segmentation": _EVERY + [
        "src/segment/mod.rs",
        "src/segment/synthseg.rs",
        "src/segment/labels.rs",
    ],
}

WIRING = {
    "stages": [
        "src/pipeline/config.rs",
        "src/pipeline/bg_removal.rs",
        "src/pipeline/inversion.rs",
        "src/pipeline/separation.rs",
    ],
    "masking": [
        "src/pipeline/config.rs",
        "src/pipeline/masking.rs",
    ],
    "segmentation": [],
}

CLASSICAL = [
    "src/bet/evolution.rs",
    "src/bet/icosphere.rs",
    "src/bet/mesh.rs",
    "src/bgremove/iharperella.rs",
    "src/bgremove/ismv.rs",
    "src/bgremove/lbv.rs",
    "src/bgremove/pdf.rs",
    "src/bgremove/resharp.rs",
    "src/bgremove/sdf.rs",
    "src/bgremove/sharp.rs",
    "src/bgremove/smv.rs",
    "src/bgremove/vsharp.rs",
    "src/crop.rs",
    "src/inversion/admm.rs",
    "src/inversion/amp_pe.rs",
    "src/inversion/cosmos.rs",
    "src/inversion/fansi.rs",
    "src/inversion/hdqsm.rs",
    "src/inversion/heidi.rs",
    "src/inversion/ilsqr.rs",
    "src/inversion/l1qsm.rs",
    "src/inversion/lsqr_qsm.rs",
    "src/inversion/medi.rs",
    "src/inversion/msmv.rs",
    "src/inversion/ndi.rs",
    "src/inversion/nltv.rs",
    "src/inversion/rts.rs",
    "src/inversion/star.rs",
    "src/inversion/sti.rs",
    "src/inversion/tfi.rs",
    "src/inversion/tgv.rs",
    "src/inversion/tikhonov.rs",
    "src/inversion/tkd.rs",
    "src/inversion/tv.rs",
    "src/inversion/whqsm.rs",
    "src/io.rs",
    "src/kernels/laplacian.rs",
    "src/kernels/smv.rs",
    "src/motion.rs",
    "src/pipeline/field_mapping.rs",
    "src/pipeline/motion.rs",
    "src/pipeline/observer.rs",
    "src/pipeline/phase_utils.rs",
    "src/pipeline/qsmart.rs",
    "src/pipeline/two_pass.rs",
    "src/priority_queue.rs",
    "src/region_grow.rs",
    "src/registration.rs",
    "src/separation/chi_sep_ilsqr.rs",
    "src/separation/chi_sep_medi.rs",
    "src/separation/decompose.rs",
    "src/separation/hc_chisep.rs",
    "src/separation/r2star_qsm.rs",
    "src/separation/wavesep.rs",
    "src/solvers/cg.rs",
    "src/swi.rs",
    "src/unwrap/bestpath.rs",
    "src/unwrap/laplacian.rs",
    "src/unwrap/mod.rs",
    "src/unwrap/prelude.rs",
    "src/unwrap/romeo.rs",
    "src/unwrap/slicewise.rs",
    "src/utils/anisotropic_diffusion.rs",
    "src/utils/complex.rs",
    "src/utils/curvature.rs",
    "src/utils/denoise.rs",
    "src/utils/epg.rs",
    "src/utils/fibermetric.rs",
    "src/utils/frangi.rs",
    "src/utils/gibbs.rs",
    "src/utils/gradient.rs",
    "src/utils/mask.rs",
    "src/utils/multi_echo.rs",
    "src/utils/ops.rs",
    "src/utils/padding.rs",
    "src/utils/qsmart.rs",
    "src/utils/special.rs",
    "src/utils/threshold.rs",
    "src/utils/union_find.rs",
    "src/utils/vasculature.rs",
    "src/utils/wavelet.rs",
]


# ---------------------------------------------------------------------------------------------
# Reading Rust source: just enough to find module paths
# ---------------------------------------------------------------------------------------------

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


def strip_comments_and_strings(text):
    """Drop comments, and blank string and char literals except plain words."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        elif c == "r" and re.match(r'r#*"', text[i:i + 8]) and not re.match(r"[A-Za-z0-9_]", text[i - 1:i] or " "):
            m = re.match(r'r(#*)"', text[i:])
            end = text.find('"' + m.group(1), i + len(m.group(0)))
            i = n if end < 0 else end + 1 + len(m.group(1))
            out.append('""')
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            # Keep a plain word (`feature = "onnx"`); blank anything that could look like a path.
            body = text[i + 1:j]
            out.append('"' + (body if re.fullmatch(r"[A-Za-z0-9_.-]*", body) else "") + '"')
            i = j + 1
        elif c == "'":
            m = re.match(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'])'", text[i:])
            if m:
                i += len(m.group(0))
                out.append("' '")
            else:  # a lifetime
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def item_end(text, start):
    """Index just past the item that starts at `start`: its balanced `{...}` or its `;`."""
    depth = 0   # of every bracket, so the `;` in `[f64; 16]` does not end a signature
    for i in range(start, len(text)):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0 and c == "}":
                return i + 1
        elif c == ";" and depth == 0:
            return i + 1
    return len(text)


ATTR = re.compile(r"#\[\s*cfg\s*\(")


def cfg_attrs(text):
    """(start, end_of_attribute, condition) for each `#[cfg(...)]`."""
    for m in ATTR.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        cond = text[m.end():i - 1]
        close = text.find("]", i)
        yield m.start(), close + 1, cond


def remove_test_items(text):
    """Drop `#[cfg(test)]` items, which the wasm build never compiles."""
    cuts = []
    for start, end, cond in cfg_attrs(text):
        if re.fullmatch(r"\s*test\s*", cond):
            cuts.append((start, item_end(text, end)))
    for start, end in reversed(cuts):
        text = text[:start] + text[end:]
    return text


def onnx_items(text):
    """The text of each item gated on the `onnx` feature (not on its absence)."""
    for start, end, cond in cfg_attrs(text):
        if 'feature = "onnx"' in re.sub(r"\s+", " ", cond).replace('feature="onnx"', 'feature = "onnx"'):
            if not re.search(r'not\s*\(\s*feature\s*=\s*"onnx"', cond):
                yield text[start:item_end(text, end)]


def expand_use_tree(tree):
    """`a::{b, c::{d as e, *}}` -> [(['a','b'], None), (['a','c','d'], 'e'), (['a','c','*'], None)]."""
    tree = re.sub(r"\s+", "", tree)
    out = []

    def walk(prefix, s):
        # split s at top-level commas
        parts, depth, cur = [], 0, ""
        for c in s:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if c == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += c
        if cur:
            parts.append(cur)
        for p in parts:
            if not p:
                continue
            brace = p.find("{")
            if brace >= 0:
                head = p[:brace].rstrip(":")
                walk(prefix + ([x for x in head.split("::") if x] if head else []), p[brace + 1:p.rfind("}")])
            else:
                out.append((prefix + p.split("::"), None))

    walk([], tree)
    return out


USE = re.compile(r"\b(pub(?:\s*\([^)]*\))?\s+)?use\s+([^;]+);")
INLINE = re.compile(r"(?<![A-Za-z0-9_:])((?:crate|super|self)(?:\s*::\s*" + IDENT + r")+)")


def use_paths(text):
    """(is_pub, segments, alias) for every `use` in `text`."""
    for m in USE.finditer(text):
        tree = m.group(2)
        # Keep `as` aliases: expand on the tree with whitespace, then recover each alias.
        for segs, _ in expand_use_tree(re.sub(r"\s+as\s+", " AS ", tree).replace(" AS ", "@AS@")):
            alias = None
            last = segs[-1]
            if "@AS@" in last:
                last, alias = last.split("@AS@")
                segs = segs[:-1] + [last]
            if segs and segs[-1] == "self":
                segs = segs[:-1]
            if segs:
                yield bool(m.group(1)), segs, alias


# ---------------------------------------------------------------------------------------------
# The module graph
# ---------------------------------------------------------------------------------------------

class Crate:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.files = sorted(str(p.relative_to(self.root)) for p in (self.root / "src").rglob("*.rs"))
        self.text = {}      # file -> source without comments, strings, test items
        self.modules = {}   # module path tuple -> (file, text of the module)
        self.macros = {}    # #[macro_export] macro -> file
        for f in self.files:
            raw = (self.root / f).read_text()
            t = remove_test_items(strip_comments_and_strings(raw))
            self.text[f] = t
            self.modules[self.modpath(f)] = (f, t)
            # Inline `mod name { ... }` blocks (lib.rs's re-export modules).
            for start, body in inline_mods(t):
                self.modules[self.modpath(f) + (start,)] = (f, body)
            if "#[macro_export]" in raw:
                for m in re.finditer(r"macro_rules!\s*(" + IDENT + ")", t):
                    self.macros.setdefault(m.group(1), f)
        self._edges = {}
        self.unresolved = set()

    @staticmethod
    def modpath(f):
        parts = Path(f).relative_to("src").with_suffix("").parts
        if parts[-1] in ("lib", "mod"):
            parts = parts[:-1]
        return tuple(parts)

    def is_module_root(self, f):
        mp = self.modpath(f)
        return any(len(k) == len(mp) + 1 and k[:-1] == mp and self.modules[k][0] != f for k in self.modules)

    def defines(self, mod, name):
        _, t = self.modules[mod]
        return re.search(r"\b(?:fn|struct|enum|trait|type|const|static|union|mod)\s+" + name + r"\b", t) or \
            re.search(r"macro_rules!\s*" + name + r"\b", t)

    def module_of(self, segs, here):
        """(module, rest): the longest module prefix of a path written in module `here`, or None
        when the path does not start in this crate (std, another crate, a local name)."""
        segs = list(segs)
        if not segs:
            return None
        if segs[0] == "crate":
            mod, segs = (), segs[1:]
        elif segs[0] in ("super", "self"):
            mod = here
            while segs and segs[0] in ("super", "self"):
                if segs[0] == "super":
                    mod = mod[:-1]
                segs = segs[1:]
        elif here + (segs[0],) in self.modules:   # a child module, by the 2018 relative path rule
            mod = here
        else:
            return None
        while segs and mod + (segs[0],) in self.modules:
            mod, segs = mod + (segs[0],), segs[1:]
        return mod, segs

    def resolve(self, segs, here, depth=0):
        """File(s) a path names, `here` being the module it is written in."""
        if depth > 12:
            return set()
        found = self.module_of(segs, here)
        if found is None:
            return set()
        mod, rest = found
        if not rest:
            return {self.modules[mod][0]}
        if rest[0] == "*":
            out = {self.modules[mod][0]}
            for _, s, _ in use_paths(top_level_with_braces(self.modules[mod][1])):
                if s[0] != "*":
                    out |= self.resolve(s, mod, depth + 1)
            return out
        return self.find_item(mod, rest[0], depth)

    def find_item(self, mod, name, depth=0):
        """The file that defines `name` as seen from module `mod`, following its `use`s."""
        f, t = self.modules[mod]
        if self.defines(mod, name):
            return {f}
        if mod == () and name in self.macros:
            return {self.macros[name]}
        globs = []
        for _, segs, alias in use_paths(top_level_with_braces(t)):
            if segs[-1] == "*":
                globs.append(segs[:-1])
            elif (alias or segs[-1]) == name:
                return self.resolve(segs, mod, depth + 1) or {f}
        for segs in globs:
            found = self.module_of(segs, mod)
            if found is None or found[1]:
                continue
            tmod = found[0]
            if self.defines(tmod, name) or any(
                    (a or s[-1]) == name for _, s, a in use_paths(top_level_with_braces(self.modules[tmod][1]))):
                return self.find_item(tmod, name, depth + 1)
        self.unresolved.add((f, name))
        return {f}

    def edges(self, f, onnx_only=False):
        """The files `f` uses. `onnx_only`: only from its `#[cfg(feature = "onnx")]` items."""
        key = (f, onnx_only)
        if key in self._edges:
            return self._edges[key]
        t = self.text[f]
        here = self.modpath(f)
        if onnx_only:
            scope = "\n".join(onnx_items(t))
        elif self.is_module_root(f):
            # A module root's `pub use`s are its re-exports, not its own dependencies: a path
            # through one is resolved to the file that defines the item (find_item).
            scope = PUB_USE.sub("", t)
        else:
            scope = t
        out = set()
        local_mods = {}
        for _, segs, alias in use_paths(scope):
            out |= self.resolve(segs, here)
            found = self.module_of(segs, here)
            if found is not None and not found[1]:   # `use crate::fft;` then `fft::fft3d(...)`
                local_mods[alias or segs[-1]] = found[0]
        for m in INLINE.finditer(scope):
            out |= self.resolve([s.strip() for s in m.group(1).split("::")], here)
        for name, mod in local_mods.items():
            for m in re.finditer(r"(?<![A-Za-z0-9_:])" + name + r"\s*::\s*(" + IDENT + ")", scope):
                out |= self.find_item(mod, m.group(1))
        for name, mf in self.macros.items():
            if re.search(r"\b" + name + r"\s*!", scope):
                out.add(mf)
        out.discard(f)
        self._edges[key] = out
        return out

    def reach(self, shard):
        """Every file the shard's ROOTS and WIRING reach, each with the file that first reached it."""
        seen = {}
        queue = deque()
        for f in ROOTS[shard]:
            seen[f] = "root"
            queue.append((f, False))
        for f in WIRING[shard]:
            if f not in seen:
                seen[f] = "wiring"
                queue.append((f, True))
        while queue:
            f, onnx_only = queue.popleft()
            if f not in self.text:
                continue
            for g in sorted(self.edges(f, onnx_only)):
                if g not in seen:
                    seen[g] = f
                    queue.append((g, False))
        return seen


PUB_USE = re.compile(r"\bpub(?:\s*\([^)]*\))?\s+use\s+[^;]+;")


def inline_mods(text):
    """(name, body) of each `mod name { ... }` at the top level of `text`."""
    depth, i = 0, 0
    while i < len(text):
        c = text[i]
        if depth == 0:
            m = re.compile(r"\bmod\s+(" + IDENT + r")\s*\{").match(text, i)
            if m and (i == 0 or not re.match(r"[A-Za-z0-9_]", text[i - 1])):
                end = item_end(text, i)
                yield m.group(1), text[m.end():end - 1]
                i = end
                continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1


def top_level_with_braces(text):
    """The `use` statements at the top level of `text` (outside every `{...}` but their own)."""
    out, depth, i = [], 0, 0
    while i < len(text):
        c = text[i]
        if depth == 0:
            m = USE.match(text, i) if text.startswith(("use", "pub"), i) else None
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Selection and check
# ---------------------------------------------------------------------------------------------

def matches(f, patterns):
    import fnmatch
    return any(fnmatch.fnmatchcase(f, p) for p in patterns)


def select(changed, crate=None):
    """(shards, reasons) for a list of changed paths."""
    crate = crate or Crate()
    reach = {s: crate.reach(s) for s in SHARDS}
    shards, reasons = set(), []
    for f in changed:
        if matches(f, ALWAYS):
            shards |= set(SHARDS)
            reasons.append(f"{f}: build, wasm test or CI input → all shards")
        elif f in NEUTRAL:
            reasons.append(f"{f}: declarations and re-exports only → none")
        elif any(f in reach[s] for s in SHARDS):
            hit = [s for s in SHARDS if f in reach[s]]
            shards |= set(hit)
            via = {s: reach[s][f] for s in hit}
            how = ", ".join(f"{s} ({'root' if v == 'root' else 'wiring' if v == 'wiring' else 'via ' + v})"
                            for s, v in via.items())
            reasons.append(f"{f}: deep-learning code → {how}")
        elif f in CLASSICAL:
            reasons.append(f"{f}: classical → none")
        elif f.startswith("src/"):
            shards |= set(SHARDS)
            reasons.append(f"{f}: not classified → all shards (add it to scripts/wasm_dl_shards.py)")
        else:
            reasons.append(f"{f}: outside the wasm build → none")
    return [s for s in SHARDS if s in shards], reasons


def check(crate=None):
    crate = crate or Crate()
    errors = []
    listed = set(NEUTRAL) | set(CLASSICAL) | {f for s in SHARDS for f in ROOTS[s] + WIRING[s]}
    for f in sorted(listed):
        if f not in crate.text:
            errors.append(f"{f} is listed but does not exist")
    reach = {s: crate.reach(s) for s in SHARDS}
    dl = set().union(*reach.values())
    for f in CLASSICAL:
        for s in SHARDS:
            if f in reach[s]:
                errors.append(f"{f} is CLASSICAL but the {s} shard reaches it via {reach[s][f]}")
    for f in NEUTRAL:
        if f not in crate.text:
            continue
        t = crate.text[f]
        # Inline `mod x { pub use ...; }` blocks are declarations too.
        body = re.sub(r"\bmod\s+" + IDENT + r"\s*\{[^{}]*\}", "", t)
        m = re.search(r"\b(fn|struct|enum|impl|trait|const|static|union|type|macro_rules!)\b", body)
        if m:
            errors.append(f"{f} is NEUTRAL but defines something ({m.group(1)}); classify it instead")
    machinery = re.compile(r"crate\s*::\s*models\b|\btract_|feature\s*=\s*\"(onnx|simd)\"")
    for f in crate.files:
        if f in dl or f in NEUTRAL:
            continue
        if machinery.search(crate.text[f]):
            errors.append(f"{f} uses the model machinery (crate::models, tract, or the onnx/simd features) "
                          "but no deep-learning shard reaches it: add it to ROOTS or WIRING")
    unclassified = [f for f in crate.files if f not in listed and f not in dl]
    return errors, unclassified, reach


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("select")
    sp.add_argument("files", nargs="*")
    sp.add_argument("--from", dest="from_file")
    sp.add_argument("--github-output", action="store_true")
    sub.add_parser("check")
    gp = sub.add_parser("graph")
    gp.add_argument("file")
    args = ap.parse_args()

    if args.cmd == "select":
        files = list(args.files)
        if args.from_file:
            files += [l.strip() for l in Path(args.from_file).read_text().splitlines() if l.strip()]
        shards, reasons = select(files)
        for r in reasons:
            print(r)
        print(json.dumps(shards))
        if args.github_output:
            with open(os.environ["GITHUB_OUTPUT"], "a") as out:
                out.write(f"wasm_dl_shards={json.dumps(shards)}\n")
    elif args.cmd == "check":
        crate = Crate()
        errors, unclassified, reach = check(crate)
        for s in SHARDS:
            print(f"{s}: {len(reach[s])} files")
        for f in unclassified:
            print(f"note: {f} is not classified, so a change to it runs every shard")
        if crate.unresolved:
            for f, name in sorted(crate.unresolved):
                print(f"note: `{name}` not found from {f}; counted as defined there")
        for e in errors:
            print(f"::error::{e}")
        sys.exit(1 if errors else 0)
    elif args.cmd == "graph":
        crate = Crate()
        for s in SHARDS:
            r = crate.reach(s)
            if args.file not in r:
                print(f"{s}: not reached")
                continue
            chain, f = [args.file], args.file
            while r[f] not in ("root", "wiring"):
                f = r[f]
                chain.append(f)
            print(f"{s}: " + " ← ".join(chain) + f" ({r[f]})")


if __name__ == "__main__":
    main()
