"""Vocabulary layer: the taxonomy.

``configs/taxonomy.yaml`` (classes, states, mask policy, verbs, tools) backs this
module; it is loaded lazily and cached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy.yaml"


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
@dataclass
class Taxonomy:
    """The unified vocabulary: classes, states, verbs, tools."""

    classes: dict[str, dict]
    verbs: dict[str, dict]
    tools: list[str]
    virtual_nodes: dict[str, dict] = field(default_factory=dict)
    on_bench_needs_geom: bool = True
    directions: list[str] = field(default_factory=list)
    #: Classes the annotation tool may create an implied instance of when the
    #: desktop references one but its log never operates on it. Empty unless
    #: ``taxonomy.yaml`` says otherwise: implying a part is a decision about the
    #: dataset, so nothing is ever implied by default.
    implied_when_referenced: list[str] = field(default_factory=list)
    #: ``view -> gold | silver | bronze``: how carefully that camera's
    #: frames are annotated at all. Every export carries it as ``tier``. It says
    #: nothing about whether a human confirmed any particular frame -- that is
    #: the separate ``verified`` field, and conflating the two is what the old
    #: single ``quality`` field did.
    view_tiers: dict[str, str] = field(default_factory=dict)
    #: ``class -> host class``: every instance of the key class is mounted on
    #: the desktop's *unique* instance of the value class and rides out of the
    #: chassis inside it. Gathered from the per-class ``host_class`` key of
    #: ``taxonomy.yaml``; empty unless a class says so.
    host_classes: dict[str, str] = field(default_factory=dict)

    # -- class queries ------------------------------------------------------
    def _defn(self, cls: str) -> dict:
        try:
            return self.classes[cls]
        except KeyError:
            pass
        try:
            return self.virtual_nodes[cls]
        except KeyError:
            raise KeyError(f"unknown taxonomy class: {cls!r}") from None

    def states_of(self, cls: str) -> list[str]:
        return list(self._defn(cls)["states"])

    def group_of(self, cls: str) -> str:
        """The class's group (``structure``, ``part``, ...), or ``""``.

        A rough statement of how deep in the machine a class sits, which is what
        lets a caller order instances the way they are physically stacked; an
        unknown class simply has no group rather than raising, since this is
        only ever used to sort.
        """
        try:
            return str(self._defn(cls).get("group") or "")
        except KeyError:
            return ""

    def default_state(self, cls: str) -> str:
        return self._defn(cls)["default_state"]

    def host_class(self, cls: str) -> Optional[str]:
        """The class whose one instance this class's instances ride on, or ``None``.

        A ``ram_latch`` is moulded into the motherboard: it is never removed by
        an action of its own, and when the board is lifted out the latch goes
        with it. Saying that here is what lets
        :func:`tda.core.graph_infer.infer_relational_fields` fill the latch's
        ``parent``/``attached`` pair, which is the mechanism already
        used for a captive screw. An unknown class simply has no host rather
        than raising, like :meth:`group_of`: callers sweep every key of an
        instance table, drafts and virtual nodes included.
        """
        return self.host_classes.get(cls)

    def needs_mask(self, cls: str, state: str, placement: str) -> bool:
        """Does this instance need its own geometry in the given placement?

        ``on_bench``  -> yes for every class but ``chassis`` (which never goes
        on the bench) and the virtual nodes (no mask, ever);
        ``elsewhere`` -> no (it is out of every view);
        ``in_chassis`` -> the per-class state table from ``taxonomy.yaml``.
        """
        if placement == "on_bench":
            if cls == "chassis" or cls in self.virtual_nodes:
                return False
            return bool(self.on_bench_needs_geom)
        if placement == "elsewhere":
            return False
        return bool(self._defn(cls)["needs_mask"].get(state, False))

    # -- verbs --------------------------------------------------------------
    def opened_or_taken_out(self) -> frozenset[str]:
        """Classes whose parts are either opened in place or taken out.

        Exactly the classes both ``open`` and ``remove`` apply to: a cover or a
        PSU / drive / card / cooler latch can swing open and stay on the
        machine, or come off and go on the pile, and only the pictures say
        which. A RAM clip or a socket lever is moulded into the board and only
        ever opens, so it is not one of them.
        """
        opens = (self.verbs.get("open") or {}).get("applies_to") or ()
        removes = (self.verbs.get("remove") or {}).get("applies_to") or ()
        return frozenset(opens) & frozenset(removes)

    def apply_verb(self, cls: str, attrs: dict, verb: str) -> Optional[tuple[str, str]]:
        """Return ``(attr, new_value)`` or ``None`` when nothing changes.

        ``None`` covers both "this verb does not apply to this class" and
        "this verb changes no state" (``reorient`` only breaks the pose).
        """
        spec = self.verbs.get(verb)
        if spec is None or cls not in spec["applies_to"]:
            return None
        effect = spec.get("effect")
        if not effect:
            return None
        attr = effect["attr"]
        if "to" in effect:
            return (attr, effect["to"])
        if "by_class" in effect:
            value = effect["by_class"].get(cls)
            return (attr, value) if value else None
        cond = effect["conditional"]
        chosen = cond["if_true"] if attrs.get(cond["if_attr"]) else cond["if_false"]
        return (attr, chosen)


def _host_classes(classes: dict[str, dict]) -> dict[str, str]:
    """The per-class ``host_class`` declarations, checked against the classes.

    Two ways to write one that can never mean anything, both raised while the
    file is read rather than discovered months later as a field nobody filled:

    * a host that is **not a class** of the taxonomy resolves to nothing for
      ever, leaving every instance of the declaring class without a parent;
    * a class naming **itself** would resolve to the instance itself on any
      desktop with exactly one of them -- a parent cycle of length one, which
      the attached cascade is no place to find out about.
    """
    hosts = {
        cls: str(defn["host_class"]).strip()
        for cls, defn in classes.items()
        if str((defn or {}).get("host_class") or "").strip()
    }
    for cls, host in sorted(hosts.items()):
        if host == cls:
            raise ValueError(
                f"taxonomy.yaml: class {cls!r} declares itself as its host_class; "
                f"an instance cannot ride on itself"
            )
        if host not in classes:
            raise ValueError(
                f"taxonomy.yaml: class {cls!r} declares host_class {host!r}, "
                f"which is not a class of the taxonomy"
            )
    return hosts


@lru_cache(maxsize=None)
def _load_taxonomy_cached(path: str) -> Taxonomy:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Taxonomy(
        classes=cfg["classes"],
        verbs=cfg["verbs"],
        tools=list(cfg["tools"]),
        virtual_nodes=cfg.get("virtual_nodes", {}),
        on_bench_needs_geom=bool(cfg.get("on_bench_needs_geom", True)),
        directions=list(cfg.get("directions", [])),
        implied_when_referenced=[
            str(c) for c in (cfg.get("implied_when_referenced") or [])
        ],
        view_tiers={str(k): str(v) for k, v in (cfg.get("view_tiers") or {}).items()},
        host_classes=_host_classes(cfg["classes"]),
    )


def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> Taxonomy:
    """Load (and cache) ``configs/taxonomy.yaml``."""
    return _load_taxonomy_cached(str(Path(path).resolve()))
