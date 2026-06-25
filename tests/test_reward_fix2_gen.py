"""FIX 2 (09-REWARD-FIX-DESIGN) — gen reward template-aware validity tests.

Verifies `_is_valid_wp_php(code)` credits well-formed Elementor/Underscore
`content_template()` completions (mix of `<?php ?>` and `<# #>`/`<%- %>`/`{{ }}`)
that plain `php -l` rejects, WITHOUT relaxing the gate for genuinely broken PHP.
Also asserts the end-to-end gen guard awards a non-zero scalar to a valid template
(was 0). Requires the `php` binary; skipped if unavailable.
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
    reason="php binary required for php -l parseability tiers",
)


def _valid(code: str) -> bool:
    from scripts.rl_rollouts import _is_valid_wp_php

    return _is_valid_wp_php(code)


# A real Elementor content_template() inside a Widget_Base subclass mixing
# <?php ?>, <# if (...) { #> control blocks, <%- view.x %> interpolation, and
# {{ }} / {{{ }}} mustache output. A directive embeds a raw `<?` (Underscore JS
# can contain comparison operators), which makes plain `php -l` open a spurious
# PHP region and FAIL — so this completion exercises the neutralization branch,
# not the fast path.
_ELEMENTOR_TEMPLATE = textwrap.dedent(
    """\
    <?php
    class My_Heading_Widget extends \\Elementor\\Widget_Base {

        protected function content_template() {
            ?>
            <div class="elementor-heading">
                <# if ( settings.title && settings.size <? 5 ) { #>
                    <h2 class="title"><%- view.getEditModel().get( settings.title ) %></h2>
                <# } else { #>
                    <span>{{ settings.fallback }}</span>
                <# } #>
                <div class="desc">{{{ settings.raw_html }}}</div>
            </div>
            <?php
        }
    }
    """
)


def test_standalone_php_still_valid():
    """`<?php function f(){}` -> True (fast path, no regression)."""
    assert _valid("<?php function f(){}") is True
    assert _valid("<?php\nfunction render( $x ) {\n    return absint( $x );\n}") is True


def test_elementor_template_valid():
    """Real content_template() with <# #>/<%- %>/{{ }} + <?php -> True (neutralized)."""
    from scripts.rl_rollouts import _is_parseable_php

    # Precondition: the raw mix is NOT plain-parseable (so the neutralization
    # branch is what makes it valid — this is not a fast-path pass).
    assert _is_parseable_php(_ELEMENTOR_TEMPLATE) is False
    assert _valid(_ELEMENTOR_TEMPLATE) is True


def test_bare_method_body_template_valid():
    """The REAL completion form: a bare method body with NO leading `<?php`.

    Regression for the verification finding: models emit `function content_template()
    { ?> ...markup... <?php }` (the body of a method already inside a PHP/class
    context) — no opening `<?php`, so `php -l` reads the head as HTML and chokes on
    the trailing `}`. The fix prepends a `<?php` opener before neutralize+lint. This
    is the actual case observed in logs/phase09_rerun/probe_weights.log (PROBE 2).
    """
    from scripts.rl_rollouts import _is_parseable_php
    from eval.output_parsers import extract_php_code

    bare = (
        "function content_template() {\n"
        "\t?>\n"
        "\t<#\n"
        "\tif ( '' === settings.selected_icon.value ) { return; }\n"
        "\tlet iconTag = 'div';\n"
        "\t#>\n"
        "\t<{{{ iconTag }}}>\n"
        "\t\t<# if ( settings.link.url !== '' ) { #>\n"
        "\t\t\t<%- view.getRenderAttributeString( 'link_url' ) %>\n"
        "\t\t<# } #>\n"
        "\t</{{{ iconTag }}}>\n"
        "\t<?php\n"
        "}"
    )
    assert _is_parseable_php(bare) is False  # raw bare body is not lintable
    assert _valid(bare) is True              # prepend-<?php + neutralize -> lints
    # And as the gen path actually sees it (after extract_php_code):
    assert _valid(extract_php_code(bare)) is True


def test_garbage_not_valid():
    """Prose / broken PHP with no template markers -> False."""
    assert _valid("This is just an explanation, here is what I'd do.") is False
    assert _valid("<?php func tion broken( {") is False


def test_template_with_broken_php_invalid():
    """Template markers present but the PHP scaffold is malformed -> False."""
    broken = textwrap.dedent(
        """\
        <?php
        class W extends \\Elementor\\Widget_Base {
            protected function content_template() {
                ?>
                <p>{{ settings.title }}</p>
                <?php
                echo ;   // malformed PHP statement survives neutralization
            // missing closing braces
        """
    )
    assert _valid(broken) is False


def test_gen_reward_nonzero_on_template(monkeypatch):
    """End-to-end gen guard: a valid template earns a non-zero scalar (was 0).

    Drives the exact guard line from collect_rollouts:
        if not _is_valid_wp_php(php): reward.scalar = 0.0
    A valid Elementor template must survive the guard with its rubric scalar intact.
    """
    from scripts.rl_rollouts import _is_valid_wp_php

    # Simulate a compute_group_rewards RewardResult with a positive composite.
    class _R:
        def __init__(self, scalar):
            self.scalar = scalar

    php_codes = [_ELEMENTOR_TEMPLATE]
    reward_results = [_R(0.72)]  # a real positive composite from the pipeline

    # The guard, verbatim:
    for i, php in enumerate(php_codes):
        if not _is_valid_wp_php(php):
            reward_results[i].scalar = 0.0

    assert reward_results[0].scalar == pytest.approx(0.72)
    assert reward_results[0].scalar > 0.0
    # And confirm the pre-fix gate (_is_parseable_php) WOULD have zeroed it:
    from scripts.rl_rollouts import _is_parseable_php

    assert _is_parseable_php(_ELEMENTOR_TEMPLATE) is False
