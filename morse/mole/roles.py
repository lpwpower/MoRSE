from __future__ import annotations

from enum import Enum


class Role(str, Enum):
	EXECUTE = "execute"
	AGGREGATE = "aggregate"


ROLE_TO_ID = {
	Role.EXECUTE: 0,
	Role.AGGREGATE: 1,
}


def role_id(role: Role) -> int:
	return int(ROLE_TO_ID[role])

