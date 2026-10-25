"""Frontend contract tests for static/index.html.

The browser is not exercised here, but the failures these guard against are all
static: an element id referenced by the render code must exist, the model name
must come from /api/config rather than being hardcoded, and the view must not
render prose explanation in place of a reconstruction.
"""

from __future__ import annotations

import io
import pathlib
import re
from collections import Counter

import pytest

from app import ai_client

ROOT = pathlib.Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return io.open(HTML_PATH, encoding="utf-8").read()


def test_model_is_not_hardcoded(html: str):
    # The badge is filled from /api/config; no model may ship as placeholder copy.
    assert "Gemma" not in html
    assert "gemma-3" not in html
    assert "flash-lite" not in html.lower()
    assert '<span id="modelTag">loading…</span>' in html


def test_model_copy_is_bound_to_config(html: str):
    assert 'id="modelFootnote"' in html
    assert '$("modelTag").textContent = c.model;' in html


def test_no_stale_model_as_a_configured_default():
    for name in ("readme.md", ".env.example"):
        text = io.open(ROOT / name, encoding="utf-8").read()
        assert "HF_MODEL_ID" not in text
        assert "SmolVLM" not in text
        assert "Qwen" not in text
    env_example = io.open(ROOT / ".env.example", encoding="utf-8").read()
    assert ai_client.DEFAULT_JARVISLABS_MODEL in env_example
    assert "JARVISLABS_MODEL=" in env_example
    assert "JARVIS_API_TOKEN=" in env_example


def test_no_local_inference_copy_survives(html: str):
    """The app is cloud-backed now; local-model promises would be false."""
    for stale in (
        "Runs locally on your own machine",
        "Running the local model",
        "Model weights not found",
        "HF_TOKEN",
        "huggingface.co/settings/tokens",
        "a local vision-language model",
    ):
        assert stale not in html, f"stale local-inference copy: {stale!r}"


def test_key_banner_points_at_jarvislabs(html: str):
    assert "JARVISLABS_API_KEY" in html
    assert "jarvislabs.ai/settings/api-keys" in html
    assert '<span id="modelTag">loading…</span>' in html


@pytest.mark.parametrize(
    "target",
    ["mmdMain", "sideMmd"],
    ids=["diagram-panel", "comparison-panel"],
)
def test_render_targets_exist_in_the_dom(html: str, target: str):
    """drawMermaid() returns silently when its target id is absent.

    A mismatch here is what left the side-by-side diagram blank.
    """
    assert f'id="{target}"' in html, f'drawMermaid target {target} has no element'
    assert f'drawMermaid("{target}"' in html, f"{target} is never rendered"


def test_mermaid_render_ids_are_unique(html: str):
    """mermaid.render() injects a temporary node under the id it is given."""
    assert 'mermaid.render(renderId, code)' in html
    assert "mermaid.render(id + \"Svg\"" not in html
    assert "++mermaidRenderCount" in html


def test_render_guards_against_a_detached_panel(html: str):
    assert "el.isConnected" in html


def test_svg_export_reads_the_rendered_node(html: str):
    assert 'id="dlSvgBtn"' in html
    assert 'document.querySelector("#mmdMain svg")' in html or "function liveSvg" in html
    assert '"image/svg+xml"' in html


def test_diagram_png_export(html: str):
    """The SVG must be made standalone before rasterising: mermaid emits a
    CSS-sized DOM fragment that an <img> refuses to load."""
    assert 'id="dlPngBtn"' in html
    assert "$('dlPngBtn').onclick" in html or '$("dlPngBtn").onclick = () => downloadDiagramPng(slug);' in html
    assert "function downloadDiagramPng" in html
    assert "function standaloneSvg" in html
    assert 'setAttribute("xmlns"' in html
    assert '"image/png"' in html
    assert 'canvas.toBlob' in html


def test_png_export_is_bounded(html: str):
    # A wide ER diagram at 2x would exceed the canvas size limit.
    assert "12000 / Math.max(width, height)" in html


def test_mermaid_uses_native_text_labels(html: str):
    """foreignObject labels are not drawn when an SVG is loaded as an <img>,
    which would silently break the PNG export."""
    assert "htmlLabels:false" in html


def test_element_ids_are_unique(html: str):
    """Duplicate ids break the getElementById wiring between panels."""
    ids = re.findall(r'\sid="([^"]+)"', html)
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    assert not dupes, f"duplicate element ids: {sorted(dupes)}"


def test_explanation_prose_is_not_rendered(html: str):
    """The reconstruction is the deliverable; prose may be exported only."""
    assert "${d.diagram_explanation" not in html
    assert "${esc(d.diagram_explanation)}" not in html


def test_sample_gallery_is_wired(html: str):
    assert "/api/samples" in html
    assert "/api/config" in html
    assert 'data-mode="${esc(s.mode)}"' in html


def test_share_image_is_rendered_deterministically(html: str):
    """The card is drawn with canvas 2D, never a model: formulas and
    identifiers must survive byte-for-byte."""
    assert "function downloadShareImage" in html
    assert "canvas.toBlob" in html
    assert '"image/png"' in html
    # No model call involved in producing the image.
    assert "generateContent" not in html
    assert "generate_content" not in html


def test_share_image_renders_every_section(html: str):
    for field in ("d.title", "d.summary", "d.notes", "d.takeaways", "d.action_items"):
        assert field in html, f"{field} missing from the share card"


def test_share_image_is_wired_to_the_export_menu(html: str):
    assert 'id="pngDl"' in html
    assert '$("pngDl").onclick = () => downloadShareImage(d, slug);' in html


def test_share_image_uses_high_dpi_scale(html: str):
    assert "scale: 2" in html
    assert "ctx.scale(CARD.scale, CARD.scale)" in html


def test_failed_runs_show_diagnostics(html: str):
    assert "function diagnosticsHtml" in html
    assert "${diagnosticsHtml(d, t)}" in html
    assert "t.raw_preview" in html
    assert "t.diagram_error" in html
    assert 'class="diag"' in html


def test_diagnostics_escape_model_output(html: str):
    assert "<pre class=\"src\">${esc(t.raw_preview)}</pre>" in html


def test_rendered_panels_escape_model_text(html: str):
    """Everything interpolated into innerHTML must go through esc()."""
    for field in ("d.title", "n.heading", "n.confidence", "a.task", "u.reason"):
        assert f"esc({field})" in html, f"{field} reaches innerHTML unescaped"
    assert "(n.content || []).map(c => `<li>${esc(c)}</li>`)" in html


def test_default_model_is_documented(html: str):
    assert ai_client.DEFAULT_MODEL in ai_client.KNOWN_MODELS


class TestSettingsScreen:
    """The API key is entered, tested and saved from inside the app."""

    def test_the_settings_button_exists(self, html: str):
        assert 'id="settingsBtn"' in html

    def test_the_key_is_entered_in_a_password_field(self, html: str):
        assert 'id="apiKeyInput"' in html
        assert 'type="password"' in html

    def test_the_key_is_never_rendered_back_into_the_page(self, html: str):
        """The key is write-only: the page shows a masked form, never the key."""
        assert 'id="keyState"' in html
        assert "api_key_masked" in html
        # No code path writes the submitted key back into the document.
        assert "$('apiKeyInput').value = key" not in html

    def test_there_is_a_connection_test(self, html: str):
        assert 'id="testKeyBtn"' in html
        assert "/api/settings/api-key/test" in html

    def test_there_is_a_save_and_a_remove(self, html: str):
        assert 'id="saveKeyBtn"' in html
        assert 'id="clearKeyBtn"' in html
        assert "/api/settings/api-key" in html

    def test_the_key_is_cleared_from_the_input_after_saving(self, html: str):
        assert '$("apiKeyInput").value = "";' in html

    def test_the_test_result_says_what_happened(self, html: str):
        assert "function keyResult" in html
        assert "keyMessage(\"ok\", keyResult(data))" in html

    def test_a_failed_analysis_offers_the_settings_screen(self, html: str):
        assert 'id="errSettingsBtn"' in html
        assert '$("errSettingsBtn").onclick = () => openSettings();' in html

    def test_a_plain_http_connection_is_flagged(self, html: str):
        """A key typed into an insecure page travels in the clear."""
        assert 'id="keyTransportNote"' in html
        assert "function warnAboutPlaintext" in html
        assert "not served over HTTPS" in html


class TestGuideAndGradeTabs:
    def test_the_guide_tab_exists(self, html: str):
        assert '["guide","Guide"' in html
        assert "function renderGuide" in html

    def test_the_grade_tab_exists(self, html: str):
        assert '["grade","Grade"' in html
        assert "function renderGrade" in html

    def test_the_grade_is_shown_next_to_the_title(self, html: str):
        assert "Grade ${esc(d.assessment.score)}/100" in html

    def test_the_guide_content_is_escaped(self, html: str):
        for field in ("g.overview", "s.detail", "c.term || c.meaning", "c.meaning"):
            assert f"esc({field})" in html, f"{field} reaches innerHTML unescaped"

    def test_the_grade_content_is_escaped(self, html: str):
        for field in ("c.criterion", "c.evidence", "a.rationale"):
            assert f"esc({field})" in html, f"{field} reaches innerHTML unescaped"


class TestChangesAreSurfaced:
    """A change to the user's own drawing must be visible, with a reason."""

    def test_deviations_are_rendered(self, html: str):
        assert "function renderReview" in html
        assert "Changes to your original idea" in html

    def test_a_deviation_shows_what_changed_and_why(self, html: str):
        assert "esc(v.change)" in html
        assert "esc(v.why)" in html

    def test_the_needs_review_tab_counts_the_changes(self, html: str):
        assert "d.uncertain_text.length + (d.deviations || []).length" in html

    def test_the_diagram_tab_points_at_the_review_tab(self, html: str):
        assert 'id="jumpReviewBtn"' in html

    def test_the_changes_are_exported(self, html: str):
        assert "## Changes to the original idea" in html