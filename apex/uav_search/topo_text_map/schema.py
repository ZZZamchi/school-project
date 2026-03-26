"""与用户约定一致的 JSON 结构（dataclass）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List


@dataclass
class Node:
    node_id: str
    semantic_description: str
    visit_count: int = 1


@dataclass
class Edge:
    from_node: str
    to_node: str
    action_taken: str
    semantic_reason: str


@dataclass
class Frontier:
    source_node: str
    direction: str
    visual_observation: str


@dataclass
class MapGraph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    unexplored_frontiers: List[Frontier] = field(default_factory=list)


@dataclass
class TopoTextMap:
    task: str
    current_location_id: str
    map_graph: MapGraph

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TopoTextMap:
        mg = d["map_graph"]
        nodes = [Node(**n) for n in mg["nodes"]]
        edges = [Edge(**e) for e in mg["edges"]]
        ufs = [Frontier(**f) for f in mg["unexplored_frontiers"]]
        return cls(
            task=d["task"],
            current_location_id=d["current_location_id"],
            map_graph=MapGraph(nodes=nodes, edges=edges, unexplored_frontiers=ufs),
        )
