"""Self-contained CRF baseline: Stage-1 RMA + released CRF model.

Bundled from the authors' prior morphological toolkit (anonymised for
review) so the repository has no external dependencies for the CRF
baseline. See LICENSES.md in this directory for per-file licenses.
"""

from .crf_tagger import CRFTagger
from .rma import ReverseMorphAnalyzer

__all__ = ["CRFTagger", "ReverseMorphAnalyzer"]
