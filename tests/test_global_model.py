from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flim_app.demo import create_demo_dataset, create_demo_irf
from flim_app.global_model import load_global_model, save_global_model
from flim_app.models import AnalysisSettings
from flim_app.processing import TrainingImageInput, analyze_dataset, train_global_model


def _training_settings(model_order: int) -> AnalysisSettings:
    return AnalysisSettings(
        method="flimlib_reconvolution",  # unused by training itself; only the native-fit fields matter
        fit_min_counts=0,
        model_order=model_order,
        native_fit_bins=29,
        fit_irf_mode=1,
        global_iterations=3,
        max_global_pixels=2_000,
    )


def test_train_global_model_across_two_images() -> None:
    items = []
    for seed in (95, 96):
        dataset = create_demo_dataset(size=8, time_bins=32, seed=seed)
        irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
        items.append(TrainingImageInput(dataset=dataset, irf=irf, name=f"seed-{seed}"))

    model = train_global_model(items, _training_settings(2), name="test-bi-model")

    assert model.model_order == 2
    assert len(model.global_taus_ns) == 2
    assert all(np.isfinite(tau) and tau > 0 for tau in model.global_taus_ns)
    assert model.global_taus_ns[0] >= model.global_taus_ns[1]  # longest-to-shortest, like everywhere else
    assert len(model.training_images) == 2
    assert model.training_selected_pixels > 0
    assert model.training_total_pixels == 8 * 8 * 2


def test_save_load_round_trip_and_apply_to_new_dataset() -> None:
    items = []
    for seed in (95, 96, 97):
        dataset = create_demo_dataset(size=8, time_bins=32, seed=seed)
        irf = create_demo_irf(dataset.y_size, dataset.time_bins, dataset.dt_ns)
        items.append(TrainingImageInput(dataset=dataset, irf=irf, name=f"seed-{seed}"))

    model = train_global_model(items, _training_settings(2), name="round-trip-model")

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.json"
        save_global_model(model, model_path)
        reloaded = load_global_model(model_path)

        assert reloaded.model_order == model.model_order
        assert reloaded.global_taus_ns == model.global_taus_ns
        assert reloaded.name == model.name
        assert len(reloaded.training_images) == len(model.training_images)

        # Apply the reloaded model to a brand-new (unseen) dataset via the
        # normal single-image pipeline, exactly like a user would from the
        # main window after loading a trained model.
        target_dataset = create_demo_dataset(size=8, time_bins=32, seed=101)
        target_irf = create_demo_irf(target_dataset.y_size, target_dataset.time_bins, target_dataset.dt_ns)
        settings = AnalysisSettings(
            method="flimlib_apply_global",
            fit_min_counts=0,
            native_fit_bins=29,
            global_model_path=str(model_path),
        )
        result = analyze_dataset(target_dataset, settings, target_irf)

        components = result.fit_diagnostics["tau_components_ns"]
        assert components.shape == (2, 8, 8)
        status = result.fit_diagnostics["fit_status"]
        successful = status == 1
        assert successful.any(), "expected at least one pixel to fit successfully"
        # Taus are fixed by the trained model, so every successful pixel must
        # show exactly the trained values (longest-to-shortest already, no
        # further sorting needed since there's nothing to sort).
        for component_index in range(2):
            values = components[component_index][successful]
            assert np.allclose(values, model.global_taus_ns[component_index])

        applied = result.metadata["native_fit"]["applied_global_model"]
        assert applied["name"] == model.name
        assert applied["training_images"] == len(model.training_images)


def main() -> None:
    tests = [
        test_train_global_model_across_two_images,
        test_save_load_round_trip_and_apply_to_new_dataset,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} global-model tests passed")


if __name__ == "__main__":
    main()
