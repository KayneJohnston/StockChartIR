"""End-to-end smoke test of the whole pipeline on the real panel."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import main
from src import data_loader as dl

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "raw" / "JST_macrohistory.xlsx").exists(),
    reason="raw JST workbook is not present",
)


@pytest.fixture(scope="module")
def pipeline_state(tmp_path_factory):
    """Run all four steps at tiny N into a temporary output tree."""
    out = tmp_path_factory.mktemp("run")
    cfg = dl.load_config(ROOT / "config.yaml")
    cfg["run"].update({
        "output_dir": str(out), "figure_dir": str(out / "figures"),
        "table_dir": str(out / "tables"), "cache_dir": str(out / "cache"),
        "processed_dir": str(out / "processed"),
    })
    cfg["data"].update({
        "jst_workbook": str(ROOT / "data" / "raw" / "JST_macrohistory.xlsx"),
        "clio_inflation": str(ROOT / "data" / "raw" / "clio_infra_inflation.csv"),
        "clio_bond_yield": str(ROOT / "data" / "raw" / "clio_infra_bond_yield.csv"),
        "emit_monthly": False,
    })
    cfg["bootstrap"].update({"n_paths": 2000, "chunk_size": 1000})
    cfg["bootstrap"]["diagnostics"]["n_paths"] = 2000
    cfg["bootstrap"]["diagnostics"]["block_length_grid"] = [2.0, 10.0]

    docs_backup = {}
    docs_dir = ROOT / "docs"
    for path in docs_dir.glob("*.md"):
        docs_backup[path] = path.read_bytes()

    state = {}
    state = main.step1_dataset(cfg)
    state = main.step2_bootstrap(cfg, state)
    state = main.step3_lifecycle(cfg, state)
    state = main.step4_report(cfg, state)
    yield cfg, state, out

    # The doc writers always target docs/; restore what was there.
    for path, blob in docs_backup.items():
        path.write_bytes(blob)


class TestPipeline:
    def test_panel_has_the_target_cross_section(self, pipeline_state) -> None:
        _, state, _ = pipeline_state
        panel = state["panel"]
        assert panel.n_countries == 38
        assert sum(1 for t in panel.tier if t == "A") == 16

    def test_every_strategy_is_simulated(self, pipeline_state) -> None:
        cfg, state, _ = pipeline_state
        assert set(state["results"]) == set(cfg["strategies"])
        for outcome in state["results"].values():
            assert outcome.n_paths == cfg["bootstrap"]["n_paths"]

    def test_headline_table_has_one_row_per_strategy(self, pipeline_state
                                                     ) -> None:
        cfg, state, _ = pipeline_state
        headline = state["headline"]
        assert len(headline) == len(cfg["strategies"])
        assert headline["cec_crra_gamma5"].gt(0).all()
        assert headline["prob_ruin"].between(0, 1).all()

    def test_all_equity_beats_the_glide_path_and_sixty_forty(self,
                                                             pipeline_state
                                                             ) -> None:
        _, state, _ = pipeline_state
        indexed = state["headline"].set_index("strategy")
        for column in ("cec_crra_gamma2", "cec_crra_gamma5", "cec_crra_gamma10"):
            assert (indexed.loc["balanced_all_equity", column]
                    > indexed.loc["target_date_fund", column])
            assert (indexed.loc["balanced_all_equity", column]
                    > indexed.loc["sixty_forty", column])
        assert (indexed.loc["balanced_all_equity", "prob_ruin"]
                < indexed.loc["target_date_fund", "prob_ruin"])

    def test_documents_are_written(self, pipeline_state) -> None:
        expected = [
            "01_country_dataset_and_sources.md",
            "02_multicountry_block_bootstrap.md",
            "03_lifecycle_utility_model.md",
            "04_replicated_results_and_tables.md",
        ]
        for name in expected:
            path = ROOT / "docs" / name
            assert path.exists() and path.stat().st_size > 2000

    def test_figures_and_tables_are_written(self, pipeline_state) -> None:
        _, _, out = pipeline_state
        assert len(list((out / "figures").glob("*.png"))) >= 11
        assert len(list((out / "tables").glob("*.csv"))) >= 15
