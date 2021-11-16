from typing import List


def extract_cfg(cfg: List[float], dofs: List[int]) -> List[float]:
    ret = []
    for d in dofs:
        ret.append(cfg[d])
    return ret
