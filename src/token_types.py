from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class FrameRole:
    role_idx: int
    role_name: str
    semantic_head_idx: Optional[int] = None

    def __repr__(self) -> str:
        return f"SYNT:{self.role_idx} SEM:{self.semantic_head_idx} // {self.role_name.upper()}"


@dataclass
class DepRelation:
    relation: Optional[str] = None
    head_idx: Optional[int] = None
    child_indices: List[int] = field(default_factory=list)

    def is_any_none(self):
        return True if any([self.relation is None, self.head_idx is None]) else False


@dataclass
class TokenFeatures:
    # Basic features
    idx: int
    text: Optional[str] = None
    lemma: Optional[str] = None
    pos: Optional[str] = None
    feat: Optional[str] = None

    # Syntactic dep features
    synt_dep: DepRelation = field(default_factory=DepRelation)

    # Semantic dep features
    sem_dep: DepRelation = field(default_factory=DepRelation)

    # SRL features
    frame_name: Optional[str] = None
    frame_roles: List[FrameRole] = field(default_factory=list)

    def to_dict(self) -> dict:
        self_dict = asdict(self)
        return self_dict

    def __repr__(self) -> str:
        # String representation of a token
        return (
            "TokenFeatures("
            + ", ".join([f"{k}={v}" for k, v in self.to_dict().items()])
            + ")"
        )


Sentence = List[TokenFeatures]
