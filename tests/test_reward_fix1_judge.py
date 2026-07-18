"""FIX 1 (09-REWARD-FIX-DESIGN) — judge reward correctness-pressure tests.

Verifies `_fix_score_from_completion(completion, original_code)` closes the
isolation hack: trivial / unrelated / gutted "fixes" no longer score ~1.0,
faithful reproduction lands at the 0.5 identity floor, and a genuine fix that
lifts the deterministic rubric scales toward 1.0. Back-compat (original=None)
preserves the legacy isolation rubric so existing call sites stay green.

Requires the `php` binary (the identity/improvement tiers depend on real
`php -l`); skipped if unavailable.
"""
from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.skipif(
    shutil.which("php") is None,
    reason="php binary required for parse-gated correctness-pressure tiers",
)


def _fix_score(completion: str, original: str | None = None) -> float:
    from scripts.rl_rollouts import _fix_score_from_completion

    return _fix_score_from_completion(completion, original)


def _fence(php: str) -> str:
    """Wrap PHP in a ```php fenced block as the policy would emit it."""
    return f"```php\n{php}\n```"


# A real buggy original: outputs $_GET unescaped (XSS). Deterministic rubric ~70.
_ORIGINAL_XSS = textwrap.dedent(
    """\
    <?php
    function render( $x ) {
        echo $_GET['q'];
    }
    """
).strip()

# A 40+ line original used for the gut-the-body anti-gutting test.
_ORIGINAL_LONG = textwrap.dedent(
    """\
    <?php
    function process_data( $input ) {
        $result = array();
        foreach ( $input as $key => $value ) {
            if ( empty( $value ) ) {
                continue;
            }
            $clean = sanitize_text_field( $value );
            if ( strlen( $clean ) > 100 ) {
                $clean = substr( $clean, 0, 100 );
            }
            $result[ $key ] = $clean;
        }
        $count = count( $result );
        if ( $count === 0 ) {
            return array();
        }
        $meta = array(
            'count'     => $count,
            'processed' => true,
            'timestamp' => time(),
        );
        $output = array(
            'data' => $result,
            'meta' => $meta,
        );
        return $output;
    }
    """
).strip()

# Genuine fix of the XSS original. NOTE: the deterministic rubric does NOT credit
# esc_html() alone for D2_security (it stays pinned at 3.0), so the spec's literal
# esc_html-only corrected yields improvement 0. We use the FULL WP-hardened fix
# (nonce + wp_unslash + sanitize + esc) which the rubric DOES credit: 70.2 -> 100.0
# (+29.8 ~ full credit). This preserves the spec's intent (genuine fix > reproduction,
# scaling toward 1.0) while being a stable assertion under the real scorer.
_CORRECTED_GENUINE_FIX = textwrap.dedent(
    """\
    <?php
    function render( $x ) {
        if ( ! isset( $_GET['q'] )
            || ! wp_verify_nonce( sanitize_key( $_GET['_wpnonce'] ?? '' ), 'q_action' ) ) {
            return;
        }
        $q = sanitize_text_field( wp_unslash( $_GET['q'] ) );
        echo esc_html( $q );
    }
    """
).strip()


def test_trivial_echo_scores_low():
    """`<?php echo 'hi';` against a buggy original -> identity fail -> 0.25."""
    score = _fix_score(_fence("<?php echo 'hi';"), _ORIGINAL_XSS)
    assert score == pytest.approx(0.25)


def test_unrelated_clean_fn_scores_low():
    """Clean unrelated function, different name -> identity (name) fail -> 0.25."""
    unrelated = "<?php\nfunction helper_format( $n ) {\n    return absint( $n );\n}"
    score = _fix_score(_fence(unrelated), _ORIGINAL_XSS)
    assert score == pytest.approx(0.25)


def test_gutted_body_scores_low():
    """Same function name, body gutted to `return 1;`, original 40+ lines.

    Passes the name gate but fails the anti-gutting length floor -> 0.25.
    """
    gutted = "<?php\nfunction process_data( $input ) {\n    return 1;\n}"
    score = _fix_score(_fence(gutted), _ORIGINAL_LONG)
    assert score == pytest.approx(0.25)


def test_gutted_body_short_original_scores_low():
    """Anti-gutting must hold for SHORT originals too (token retention, not length).

    Regression for the verification finding: a length-ratio floor let a gut-to-
    `return 1;` of a SHORT function stay above 0.5*len -> passed identity -> trivial
    clean rubric (100) -> ~1.0. Token retention catches it regardless of length:
    the gutted body drops the original's $_GET/echo/$q tokens.
    """
    orig_short = "<?php\nfunction render( $x ) {\n  $q = $_GET['q'];\n  echo $q;\n}"
    gutted = "<?php\nfunction render( $x ) { return 1; }"
    assert _fix_score(_fence(gutted), orig_short) == pytest.approx(0.25)


def test_faithful_reproduction_scores_mid():
    """corrected == original (bug intact) -> identity ok, delta 0 -> ~0.5."""
    score = _fix_score(_fence(_ORIGINAL_XSS), _ORIGINAL_XSS)
    assert score == pytest.approx(0.5, abs=1e-6)


def test_genuine_fix_scores_high():
    """Genuine WP-hardened fix lifts the rubric -> > reproduction, toward 1.0."""
    reproduction = _fix_score(_fence(_ORIGINAL_XSS), _ORIGINAL_XSS)
    fixed = _fix_score(_fence(_CORRECTED_GENUINE_FIX), _ORIGINAL_XSS)
    assert fixed > 0.5
    assert fixed > reproduction + 0.05  # meaningfully higher than reproduction
    assert fixed >= 0.6  # spec V1 hack-probe gate


def test_backcompat_no_original():
    """original_code=None -> legacy isolation rubric.overall/100 (existing tests green)."""
    from scripts.reward_pipeline import _extract_verifiable_signals

    clean = "<?php\nfunction f() {\n    return 1;\n}"
    legacy = _fix_score(_fence(clean), None)
    expected = float(_extract_verifiable_signals(clean).overall) / 100.0
    assert legacy == pytest.approx(expected)


def test_prose_only_scores_partial():
    """Non-empty unparseable (prose) -> 0.25 (tier preserved)."""
    score = _fix_score("This code looks fine to me, no changes needed.", _ORIGINAL_XSS)
    assert score == pytest.approx(0.25)


def test_empty_scores_zero():
    """Truly empty / no corrected block -> 0.0 (low extreme preserved)."""
    assert _fix_score("", _ORIGINAL_XSS) == pytest.approx(0.0)
    assert _fix_score("   ", _ORIGINAL_XSS) == pytest.approx(0.0)


def test_frac_mid_preserved():
    """A mixed batch yields >= 3 distinct score values (SC1 frac_mid gate parity)."""
    batch = [
        _fix_score("", _ORIGINAL_XSS),  # 0.0
        _fix_score(_fence("<?php echo 'hi';"), _ORIGINAL_XSS),  # 0.25
        _fix_score(_fence(_ORIGINAL_XSS), _ORIGINAL_XSS),  # 0.5
        _fix_score(_fence(_CORRECTED_GENUINE_FIX), _ORIGINAL_XSS),  # ~1.0
    ]
    distinct = {round(s, 4) for s in batch}
    assert len(distinct) >= 3
