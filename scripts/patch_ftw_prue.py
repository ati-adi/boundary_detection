#!/usr/bin/env python3
"""
Patch installed ftw-tools so FTW PRUE (v3) checkpoints load for inference.

PRUE uses loss='logcoshdice', which ftw-tools 1.4.3 (PyPI, Python 3.11) rejects.
Git main requires Python ≥3.12. This patch aliases logcoshdice → JaccardLoss
placeholder (criterion unused at inference; avoids CE weight state_dict keys).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = 'loss in ("logcoshdice", "log_cosh_dice", "dice")'


def patch() -> bool:
    import ftw.trainers as trainers

    path = Path(trainers.__file__)
    text = path.read_text()
    if MARKER in text and "JaccardLoss" in text[text.find(MARKER) : text.find(MARKER) + 400]:
        print(f"Already patched: {path}")
        return False

    # Replace any existing logcoshdice branch or inject before final else
    branch = '''
        elif loss in ("logcoshdice", "log_cosh_dice", "dice"):
            # PRUE checkpoints (ftw-baselines v3) use logcoshdice. ftw-tools 1.4.3
            # does not ship that loss; for inference the criterion is unused.
            # Use JaccardLoss (no state_dict keys) as a load-time placeholder.
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
'''

    if MARKER in text:
        print(f"Already patched: {path}")
        return False

    old_else = '''        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )'''
    new_else = branch + '''        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard', 'focal' or 'logcoshdice' loss."
            )'''
    if old_else not in text:
        # Already has an older CE-based patch; upgrade it in place
        if "logcoshdice" in text:
            text2 = text.replace(
                """self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )""",
                """self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )""",
                1,
            )
            if text2 != text and "logcoshdice" in text2:
                path.write_text(text2)
                print(f"Upgraded CE placeholder to JaccardLoss: {path}")
                return True
        print(f"Unexpected trainers.py layout at {path}; cannot patch automatically")
        sys.exit(1)
    path.write_text(text.replace(old_else, new_else, 1))
    print(f"Patched: {path}")
    return True


if __name__ == "__main__":
    patch()
