from typing import Any, List


class CircularList:
    def __init__(self, base_list: List[Any]) -> None:
        self._list = base_list

    def __getitem__(self, i: int):
        return self._list[i % len(self._list)]
