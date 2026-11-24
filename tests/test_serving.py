"""End-to-end tests for Model Serving and Model Monitoring.

The suite builds the real FastAPI application over a temporary model package, so the
catalog introspection, the saved-model loader and the child-process inference run
exactly as in production.  ``httpx.ASGITransport`` is used instead of
``starlette.TestClient`` because the installed httpx version is newer than the one
expected by Starlette 0.35.

Run with::

    python -m pytest tests/test_serving.py -v
"""

from __future__ import annotations

import asyncio
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVER_DIR))

import monitoring_core  # noqa: E402
import server_core as core  # noqa: E402
import server_fastapi  # noqa: E402
import serving_api  # noqa: E402
import serving_core  # noqa: E402

PROBE = '''
"""Temporary model package used by the serving tests."""


class Calculator:
    """A small stateful probe model."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def add(self, amount: int = 1) -> int:
        """Add ``amount`` to the current value."""
        self.value += amount
        return self.value

    def predict(self, amount: int = 1) -> int:
        """Predict by adding ``amount`` to the current value."""
        return self.value + amount

    def fit(self, amount: int = 1) -> int:
        """Training methods are not deployable."""
        self.value += amount
        return self.value

    @staticmethod
    def load(filename: str = "./calculator") -> "Calculator":
        """Load a pickled calculator."""
        import pickle

        with open(filename, "rb") as handle:
            return pickle.load(handle)


class BrokenLoader:
    """Model whose declared loader rejects a bare pickle file."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def predict(self, amount: int = 1) -> int:
        """Predict by adding ``amount`` to the current value."""
        return self.value + amount

    @staticmethod
    def load(filename: str = "./broken") -> "BrokenLoader":
        """Simulate a loader that requires a config.json folder."""
        raise RuntimeError("config.json is required by this loader")
'''


def make_app(tmp_path: Path, monkeypatch):
    """Create an application backed by an ephemeral model package and store."""
    package = tmp_path / "pkg"
    models = package / "models"
    models.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (models / "__init__.py").write_text("", encoding="utf-8")
    (models / "probe.py").write_text(PROBE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))

    settings = server_fastapi.ServerSettings(models)
    assert any(model["id"] == "Calculator" for model in settings.catalog)
    settings.serving.shutdown()
    settings.serving = serving_core.ServingStore(tmp_path / "store")
    app = server_fastapi.create_app(settings=settings)
    return app, settings


def run(scenario):
    """Run one async scenario factory inside a fresh event loop."""
    asyncio.run(scenario())


async def wait_running(client: httpx.AsyncClient, endpoint_id: str) -> dict:
    for _ in range(100):
        response = await client.get(f"/api/serving/endpoints/{endpoint_id}")
        assert response.status_code == 200, response.text
        data = response.json()
        if data["status"] != "deploying":
            return data
        await asyncio.sleep(0.05)
    raise AssertionError("the deployment did not finish in time")


def deployment_payload(**overrides):
    payload = {
        "model_id": "Calculator",
        "method": "predict",
        "endpoint_name": "calculator-predict",
        "parameters": {"value": 10},
        "config": {"timeout_seconds": 30, "max_concurrency": 2},
        "load": {"enabled": False},
    }
    payload.update(overrides)
    return payload


def test_serving_and_monitoring_end_to_end(tmp_path, monkeypatch):
    """Deploy, infer, document and monitor one endpoint."""
    app, settings = make_app(tmp_path, monkeypatch)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for page in ("/serving", "/monitoring", "/serving/calculator-predict"):
                response = await client.get(page)
                assert response.status_code == 200, (page, response.text)
                assert "text/html" in response.headers["content-type"]
            for asset in ("/serving.js", "/monitoring.js", "/charts.js", "/platform.css"):
                response = await client.get(asset)
                assert response.status_code == 200, (asset, response.text)

            openapi = await client.get("/api/openapi.json")
            assert openapi.status_code == 200
            assert "/api/serving/endpoints/{endpoint_id}/predict" in openapi.text
            assert "/api/monitoring/overview" in openapi.text

            validation = await client.post("/api/serving/validate", json=deployment_payload())
            assert validation.status_code == 200, validation.text
            assert validation.json()["valid"] is True, validation.json()

            created = await client.post("/api/serving/endpoints", json=deployment_payload())
            assert created.status_code == 202, created.text
            endpoint = await wait_running(client, "calculator-predict")
            assert endpoint["status"] == "running", endpoint
            assert endpoint["method"] == "predict"
            assert endpoint["endpoint_url"].endswith(
                "/api/serving/endpoints/calculator-predict/predict"
            )

            inference = await client.post(
                "/api/serving/endpoints/calculator-predict/predict",
                json={"input": 5},
            )
            assert inference.status_code == 200, inference.text
            body = inference.json()
            assert body["prediction"] == 15, body
            assert body["request_id"]

            docs = (await client.get("/api/serving/endpoints/calculator-predict/docs")).json()[
                "docs"
            ]
            assert docs["method"]["name"] == "predict"
            assert docs["method"]["inference_like"] is True
            assert {sample["language"] for sample in docs["samples"]} == {
                "python",
                "bash",
                "javascript",
                "http",
            }
            assert "requests.post" in docs["samples"][0]["code"]

            overview = (await client.get("/api/monitoring/overview?range=24h")).json()
            assert overview["totals"]["requests"] >= 1
            assert overview["totals"]["successful"] >= 1
            assert overview["kpis"]["total_requests"]["value"] >= 1
            assert overview["series"]["activity"]
            assert any(
                item["endpoint_id"] == "calculator-predict"
                for item in overview["breakdown"]["endpoints"]
            )

            detail = (
                await client.get("/api/monitoring/endpoints/calculator-predict?range=24h")
            ).json()
            assert detail["totals"]["requests"] >= 1
            assert detail["recent"][0]["status"] == "success"
            assert detail["deployment"]["id"] == "calculator-predict"

    try:
        run(scenario)
    finally:
        settings.serving.shutdown()


def test_lifecycle_saved_model_and_monitoring_errors(tmp_path, monkeypatch):
    """Stop/restart/delete, saved-model loading and failed-request accounting."""
    app, settings = make_app(tmp_path, monkeypatch)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            await client.post("/api/serving/endpoints", json=deployment_payload())
            assert (await wait_running(client, "calculator-predict"))["status"] == "running"

            stopped = await client.post("/api/serving/endpoints/calculator-predict/stop")
            assert stopped.status_code == 200, stopped.text
            blocked = await client.post(
                "/api/serving/endpoints/calculator-predict/predict",
                json={"input": 1},
            )
            assert blocked.status_code == 409, blocked.text

            restarted = await client.post("/api/serving/endpoints/calculator-predict/restart")
            assert restarted.status_code == 202, restarted.text
            assert (await wait_running(client, "calculator-predict"))["status"] == "running"

            probe = __import__("pkg.models.probe", fromlist=["Calculator"])
            artifact = tmp_path / "calculator.pkl"
            with open(artifact, "wb") as handle:
                pickle.dump(probe.Calculator(value=100), handle)
            loaded_payload = deployment_payload(
                endpoint_name="calculator-loaded",
                load={"enabled": True, "loader": "load", "path": str(artifact)},
            )
            loaded_created = await client.post("/api/serving/endpoints", json=loaded_payload)
            assert loaded_created.status_code == 202, loaded_created.text
            loaded = await wait_running(client, "calculator-loaded")
            assert loaded["status"] == "running", loaded
            assert loaded["loaded_artifact"]["type"] == "Calculator", loaded
            loaded_inference = await client.post(
                "/api/serving/endpoints/calculator-loaded/predict",
                json={"input": 5},
            )
            assert loaded_inference.status_code == 200, loaded_inference.text
            assert loaded_inference.json()["prediction"] == 105

            invalid_arguments = await client.post(
                "/api/serving/endpoints/calculator-predict/predict",
                json={"inputs": {"unknown": 1}},
            )
            assert invalid_arguments.status_code in {400, 422}, invalid_arguments.text

            overview = (await client.get("/api/monitoring/overview?range=24h")).json()
            assert overview["totals"]["failed"] >= 2
            assert overview["breakdown"]["errors"]["total"] >= 2

            now = time.time()
            custom = await client.get(
                "/api/monitoring/overview?range=custom" f"&start={now - 3600}&end={now + 60}"
            )
            assert custom.status_code == 200, custom.text
            assert custom.json()["range"]["key"] == "custom"

            deleted = await client.delete("/api/serving/endpoints/calculator-predict")
            assert deleted.status_code == 200, deleted.text
            missing = await client.get("/api/serving/endpoints/calculator-predict")
            assert missing.status_code == 404, missing.text
            await client.delete("/api/serving/endpoints/calculator-loaded")

    try:
        run(scenario)
    finally:
        settings.serving.shutdown()


def test_bare_pickle_loads_when_declared_loader_rejects_it(tmp_path, monkeypatch):
    """A .pkl without config.json works even if the catalog loader needs config."""
    app, settings = make_app(tmp_path, monkeypatch)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            probe = __import__("pkg.models.probe", fromlist=["BrokenLoader"])
            artifact = tmp_path / "broken.pkl"
            with open(artifact, "wb") as handle:
                pickle.dump(probe.BrokenLoader(value=40), handle)
            payload = deployment_payload(
                model_id="BrokenLoader",
                endpoint_name="broken-loaded",
                method="predict",
                parameters={"value": 40},
                load={"enabled": True, "loader": "load", "path": str(artifact)},
            )
            created = await client.post("/api/serving/endpoints", json=payload)
            assert created.status_code == 202, created.text
            loaded = await wait_running(client, "broken-loaded")
            assert loaded["status"] == "running", loaded
            assert loaded["loaded_artifact"]["type"] == "BrokenLoader"
            prediction = await client.post(
                "/api/serving/endpoints/broken-loaded/predict", json={"input": 2}
            )
            assert prediction.status_code == 200, prediction.text
            assert prediction.json()["prediction"] == 42
            await client.delete("/api/serving/endpoints/broken-loaded")

    try:
        run(scenario)
    finally:
        settings.serving.shutdown()


def test_deployment_validation_rejects_bad_configuration(tmp_path, monkeypatch):
    """Unknown methods, training methods and missing constructor arguments."""
    app, settings = make_app(tmp_path, monkeypatch)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for payload in (
                deployment_payload(method="does_not_exist"),
                deployment_payload(method="fit"),
                deployment_payload(config={"timeout_seconds": 0}),
                deployment_payload(load={"enabled": True, "path": ""}),
                deployment_payload(endpoint_name="bad name!"),
            ):
                response = await client.post("/api/serving/validate", json=payload)
                assert response.status_code == 200, response.text
                assert response.json()["valid"] is False, response.json()

            invalid = await client.post(
                "/api/serving/endpoints",
                json=deployment_payload(config={"timeout_seconds": 0}),
            )
            assert invalid.status_code in {400, 422}, invalid.text

    try:
        run(scenario)
    finally:
        settings.serving.shutdown()


def test_monitoring_core_aggregates_the_event_log(tmp_path):
    """Percentiles, buckets and error groups are computed from real events."""
    now = time.time()
    events = [
        {
            "id": "a",
            "timestamp": now - 120,
            "at": serving_core.iso(),
            "endpoint_id": "ep",
            "endpoint_name": "Endpoint",
            "model_id": "M",
            "model_name": "Model",
            "method": "predict",
            "status": "success",
            "http_status": 200,
            "latency_ms": 10.0,
        },
        {
            "id": "b",
            "timestamp": now - 60,
            "at": serving_core.iso(),
            "endpoint_id": "ep",
            "endpoint_name": "Endpoint",
            "model_id": "M",
            "model_name": "Model",
            "method": "predict",
            "status": "error",
            "http_status": 422,
            "latency_ms": 30.0,
            "error_type": "InvocationError",
            "error_message": "boom",
        },
    ]
    window = monitoring_core.parse_range("1h", now=now)
    totals = monitoring_core.aggregate(events, window["minutes"])
    assert totals["requests"] == 2
    assert totals["successful"] == 1
    assert totals["failed"] == 1
    assert totals["error_rate"] == 50.0
    assert totals["avg_latency_ms"] == 20.0

    buckets = monitoring_core.bucketize(
        events, window["start"], window["end"], window["bucket_seconds"]
    )
    assert len(buckets) >= 1
    assert sum(point["requests"] for point in buckets) == 2

    errors = monitoring_core.error_breakdown(events)
    assert errors["total"] == 1
    assert errors["top_messages"][0]["message"] == "boom"

    deployments = [{"id": "ep", "name": "Endpoint", "model_id": "M", "status": "running"}]
    payload = monitoring_core.overview(events, deployments, window)
    assert payload["kpis"]["total_requests"]["value"] == 2
    assert payload["deployments"]["active_endpoints"] == 1
    assert payload["resources"]["per_endpoint"] is False

    index = monitoring_core.endpoint_index(deployments, events)
    assert index[0]["id"] == "ep"
    assert index[0]["exists"] is True


def test_loader_module_survives_the_serving_inference_payload():
    """AutoClassifier's loader lives in a private module; predictions need it."""
    model = core.MODELS_BY_ID["AutoClassifier"]
    load = serving_core._normalise_load(
        {"enabled": True, "loader": "load", "path": "/tmp/model"}, {}, model
    )
    assert load["module"] == "likelihood.models.deep._autoencoders"
    invoke_load = serving_core._invoke_load(load)
    assert invoke_load["module"] == "likelihood.models.deep._autoencoders"
    assert invoke_load["path_parameter"] == "filepath"


def test_example_payload_uses_model_input_shape_for_new_and_loaded(tmp_path):
    """The Probar example must use the actual feature count in both modes."""
    model = core.MODELS_BY_ID["AutoClassifier"]
    method = core.find_method(model, "call")

    new_fields = core.method_fields(method, {"input_shape_parm": 5})
    new_example = serving_core.method_example(new_fields)
    assert len(new_example["input"][0]) == 5

    artifact = tmp_path / "trained"
    artifact.mkdir()
    (artifact / "artifact.json").write_text(
        json.dumps({"parameters": {"input_shape_parm": 7, "num_classes": 2}}),
        encoding="utf-8",
    )
    load = serving_core._normalise_load(
        {"enabled": True, "loader": "load", "path": str(artifact)}, {}, model
    )
    loaded_fields = core.method_fields(
        method,
        serving_core.example_parameters(
            {"parameters": {}, "loaded_artifact": {"attribute_values": load["parameters"]}},
            model,
        ),
    )
    loaded_example = serving_core.method_example(loaded_fields)
    assert len(loaded_example["input"][0]) == 7

    # A normal (non-training) Keras save only has config.json.
    config_dir = tmp_path / "native"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"input_shape_parm": 3, "num_classes": 2}), encoding="utf-8"
    )
    config_load = serving_core._normalise_load(
        {"enabled": True, "loader": "load", "path": str(config_dir)}, {}, model
    )
    assert config_load["parameters"]["input_shape_parm"] == 3

    # A .keras archive does not need an adjacent config.json; Keras embeds it.
    keras_path = tmp_path / "model.keras"
    with zipfile.ZipFile(keras_path, "w") as archive:
        archive.writestr(
            "config.json",
            json.dumps(
                {
                    "class_name": "AutoClassifier",
                    "config": {"input_shape_parm": 6, "num_classes": 2},
                }
            ),
        )
    keras_load = serving_core._normalise_load(
        {"enabled": True, "loader": "load", "path": str(keras_path)}, {}, model
    )
    assert keras_load["parameters"]["input_shape_parm"] == 6

    # Bare .pkl paths are valid even when no config/metadata file exists.
    pickle_path = tmp_path / "model.pkl"
    pickle_path.write_bytes(b"not-a-real-pickle")
    pickle_load = serving_core._normalise_load(
        {"enabled": True, "loader": "load", "path": str(pickle_path)}, {}, model
    )
    assert pickle_load["enabled"] is True
    assert pickle_load["path"].endswith("model.pkl")

    deployment = {
        "id": "ep-loaded",
        "name": "ep-loaded",
        "model_id": "ca-loaded",
        "model_type_id": "AutoClassifier",
        "method": "call",
        "parameters": {"input_shape_parm": 2},
        "loaded_artifact": {"attribute_values": {"input_shape_parm": 4}},
    }
    docs = serving_core.build_docs(deployment, model, method)
    assert len(docs["payload_example"]["input"][0]) == 4

    # The API projection used by the Probar panel must agree with docs.
    parameters = serving_api._example_parameters(model, deployment)
    assert len(serving_core.method_example(core.method_fields(method, parameters))["input"][0]) == 4

    # The wizard preview must use constructor values, not catalog defaults.
    preview = serving_core.ServingStore(tmp_path / "preview-store")
    try:
        params = core.constructor_samples(model)
        params["input_shape_parm"] = 5
        result = preview.preview(
            model,
            {
                "endpoint_name": "auto-preview",
                "method": "call",
                "parameters": params,
                "load": {"enabled": False},
            },
        )
        assert result["valid"] is True
        assert len(result["payload_example"]["input"][0]) == 5
    finally:
        preview.shutdown()


def test_inference_method_policy():
    """Training/building methods are rejected, inference methods are accepted."""
    assert serving_core.is_inference_method("predict") is True
    assert serving_core.is_inference_method("forward") is True
    assert serving_core.is_inference_method("transform") is True
    assert serving_core.is_inference_method("fit") is False
    assert serving_core.is_inference_method("fit_transform") is False
    assert serving_core.is_inference_method("build") is False
    assert serving_core.is_inference_method("train_on_batch") is False


def test_unique_model_instances_reuse_and_metric_cleanup(tmp_path, monkeypatch):
    """Two instances, a reused instance, manual purge and automatic purge."""
    app, settings = make_app(tmp_path, monkeypatch)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

            async def deploy(name):
                response = await client.post(
                    "/api/serving/endpoints",
                    json=deployment_payload(endpoint_name=name),
                )
                assert response.status_code == 202, response.text
                return await wait_running(client, name)

            first = await deploy("calc-a")
            second = await deploy("calc-b")
            assert first["model_id"] != second["model_id"]
            assert first["model_id"].startswith("mdl-")
            assert first["deployment_id"] != second["deployment_id"]
            assert first["model_type_id"] == second["model_type_id"] == "Calculator"

            instances = (await client.get("/api/serving/models")).json()["models"]
            assert {first["model_id"], second["model_id"]} <= {
                model["model_id"] for model in instances
            }
            standalone = await client.post(
                "/api/serving/models",
                json={
                    "model_type_id": "Calculator",
                    "name": "standalone-calculator",
                    "parameters": {"value": 7},
                },
            )
            assert standalone.status_code == 201, standalone.text
            standalone_model = standalone.json()["model"]
            assert standalone_model["model_id"].startswith("mdl-")
            assert (
                await client.get(f"/api/serving/models/{standalone_model['model_id']}")
            ).status_code == 200

            # A third deployment reuses the first model instance.
            reuse = deployment_payload(endpoint_name="calc-c")
            reuse["model_id"] = first["model_id"]
            reuse["model_type_id"] = "Calculator"
            reused = await client.post("/api/serving/endpoints", json=reuse)
            assert reused.status_code == 202, reused.text
            third = await wait_running(client, "calc-c")
            assert third["model_id"] == first["model_id"]
            assert third["deployment_id"] != first["deployment_id"]

            for endpoint_id in ("calc-a", "calc-b", "calc-c"):
                response = await client.post(
                    f"/api/serving/endpoints/{endpoint_id}/predict",
                    json={"input": 1},
                )
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["deployment_id"] == endpoint_id

            events_a = (
                await client.get("/api/monitoring/events?range=24h&endpoint_id=calc-a")
            ).json()["events"]
            assert events_a[0]["deployment_id"] == "calc-a"
            assert events_a[0]["model_id"] == first["model_id"]
            assert events_a[0]["model_type_id"] == "Calculator"

            models = (await client.get("/api/monitoring/models?range=24h")).json()["models"]
            by_model = {row["model_id"]: row for row in models}
            assert by_model[first["model_id"]]["deployments"] == ["calc-a", "calc-c"]
            assert by_model[second["model_id"]]["deployments"] == ["calc-b"]

            cleared = await client.delete("/api/serving/endpoints/calc-a/metrics")
            assert cleared.status_code == 200, cleared.text
            assert cleared.json()["metrics_deleted"] >= 1

            remaining_a = (
                await client.get("/api/monitoring/events?range=24h&endpoint_id=calc-a")
            ).json()
            remaining_b = (
                await client.get("/api/monitoring/events?range=24h&endpoint_id=calc-b")
            ).json()
            remaining_c = (
                await client.get("/api/monitoring/events?range=24h&endpoint_id=calc-c")
            ).json()
            assert remaining_a["total"] == 0
            assert remaining_b["total"] >= 1
            assert remaining_c["total"] >= 1

            # Deleting a deployment removes only its metrics; the model instance
            # and the other deployments of the same model stay intact.
            deleted = await client.delete("/api/serving/endpoints/calc-b")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["deployment_id"] == "calc-b"
            assert deleted.json()["model_id"] == second["model_id"]
            assert deleted.json()["metrics_deleted"] >= 1

            assert (await client.get("/api/monitoring/events?range=24h&endpoint_id=calc-b")).json()[
                "total"
            ] == 0
            assert (await client.get("/api/monitoring/events?range=24h&endpoint_id=calc-c")).json()[
                "total"
            ] >= 1
            assert (await client.get(f"/api/serving/models/{first['model_id']}")).status_code == 200
            assert (
                await client.get(f"/api/serving/models/{second['model_id']}")
            ).status_code == 200
            assert (await client.get("/api/models/Calculator")).status_code == 200

            # Deleting the last deployment of an instance keeps the instance too.
            await client.delete("/api/serving/endpoints/calc-a")
            assert (await client.get("/api/monitoring/events?range=24h&endpoint_id=calc-c")).json()[
                "total"
            ] >= 1
            assert (await client.get(f"/api/serving/models/{first['model_id']}")).status_code == 200
            await client.delete("/api/serving/endpoints/calc-c")

    try:
        run(scenario)
    finally:
        settings.serving.shutdown()


def test_metric_cleanup_and_deletion_survive_a_restart(tmp_path):
    """Manual purge and automatic purge must be durable and model-safe."""
    store_dir = tmp_path / "persisted"
    store_dir.mkdir()
    now = time.time()

    def deployment(identifier: str, model_id: str) -> dict:
        return {
            "id": identifier,
            "deployment_id": identifier,
            "name": identifier,
            "model_id": model_id,
            "model_instance_id": model_id,
            "model_type_id": "Calculator",
            "catalog_model_id": "Calculator",
            "model_name": "Calculator",
            "model_title": "Calculator",
            "method": "predict",
            "method_signature": "predict(amount: int = 1) -> int",
            "parameters": {"value": 1},
            "load": {"enabled": False},
            "config": {"timeout_seconds": 30, "max_concurrency": 1},
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "log": [],
        }

    def event(identifier: str, model_id: str, status: str) -> dict:
        return {
            "id": f"{identifier}-{status}",
            "timestamp": now,
            "at": serving_core.iso(),
            "endpoint_id": identifier,
            "deployment_id": identifier,
            "endpoint_name": identifier,
            "model_id": model_id,
            "model_instance_id": model_id,
            "model_type_id": "Calculator",
            "catalog_model_id": "Calculator",
            "model_name": "Calculator",
            "method": "predict",
            "status": status,
            "http_status": 200 if status == "success" else 500,
            "latency_ms": 2.0,
        }

    (store_dir / "deployments.json").write_text(
        json.dumps(
            {
                "version": serving_core.STORE_VERSION,
                "deployments": [
                    deployment("dep-a", "mdl-a"),
                    deployment("dep-b", "mdl-b"),
                ],
            }
        ),
        encoding="utf-8",
    )
    (store_dir / "models.json").write_text(
        json.dumps(
            {
                "version": serving_core.STORE_VERSION,
                "models": [
                    {
                        "model_id": "mdl-a",
                        "model_type_id": "Calculator",
                        "name": "instance a",
                        "parameters": {"value": 1},
                        "load": {"enabled": False},
                    },
                    {
                        "model_id": "mdl-b",
                        "model_type_id": "Calculator",
                        "name": "instance b",
                        "parameters": {"value": 2},
                        "load": {"enabled": False},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (store_dir / "events.jsonl").write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                event("dep-a", "mdl-a", "success"),
                event("dep-a", "mdl-a", "error"),
                event("dep-b", "mdl-b", "success"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    store = serving_core.ServingStore(store_dir)
    try:
        assert len(store.events) == 3
        cleaned = store.clear_metrics("dep-a")
        assert cleaned["deployment_id"] == "dep-a"
        assert cleaned["model_id"] == "mdl-a"
        assert cleaned["metrics_deleted"] == 2
        assert store.get("dep-a")["model_id"] == "mdl-a"
        assert store.get_model("mdl-a")["model_type_id"] == "Calculator"
        assert [item["deployment_id"] for item in store.events.events()] == ["dep-b"]

        removed = store.delete("dep-b")
        assert removed["deployment_id"] == "dep-b"
        assert removed["metrics_deleted"] == 1
        # Deleting one deployment never affects the sibling deployment/model.
        assert [item["id"] for item in store.list()] == ["dep-a"]
        assert store.get_model("mdl-a")["model_id"] == "mdl-a"
        assert store.get_model("mdl-b")["model_type_id"] == "Calculator"
        assert len(store.events) == 0

        # The last deployment can be deleted without touching the instances.
        removed = store.delete("dep-a")
        assert removed["metrics_deleted"] == 0
        assert store.list() == []
        assert store.get_model("mdl-a")["model_id"] == "mdl-a"
    finally:
        store.shutdown()

    reopened = serving_core.ServingStore(store_dir)
    try:
        assert reopened.list() == []
        assert len(reopened.events) == 0
        assert reopened.get_model("mdl-a")["name"] == "instance a"
        assert reopened.get_model("mdl-b")["name"] == "instance b"
    finally:
        reopened.shutdown()


def test_legacy_store_is_migrated_to_unique_instances(tmp_path):
    """Version-1 deployments/events receive stable unique model ids."""
    store_dir = tmp_path / "legacy"
    store_dir.mkdir()
    now = time.time()
    (store_dir / "deployments.json").write_text(
        json.dumps(
            {
                "version": 1,
                "deployments": [
                    {
                        "id": "old-one",
                        "name": "old-one",
                        "model_id": "Calculator",
                        "model_name": "Calculator",
                        "method": "predict",
                        "parameters": {"value": 1},
                        "load": {"enabled": False},
                        "config": {"timeout_seconds": 30, "max_concurrency": 1},
                        "status": "running",
                        "created_at": "2026-01-01T00:00:00Z",
                        "log": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (store_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "id": "old-event",
                "timestamp": now,
                "at": serving_core.iso(),
                "endpoint_id": "old-one",
                "endpoint_name": "old-one",
                "model_id": "Calculator",
                "model_name": "Calculator",
                "status": "success",
                "http_status": 200,
                "latency_ms": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = serving_core.ServingStore(store_dir)
    try:
        deployment = store.get("old-one")
        assert deployment["model_id"].startswith("mdl-")
        assert deployment["model_id"] != "Calculator"
        assert deployment["model_type_id"] == "Calculator"
        assert deployment["deployment_id"] == "old-one"
        assert store.get_model(deployment["model_id"])["model_type_id"] == "Calculator"
        event = store.events.events()[0]
        assert event["model_id"] == deployment["model_id"]
        assert event["model_type_id"] == "Calculator"
        assert event["deployment_id"] == "old-one"
        # Reopening the migrated store must be stable (idempotent migration).
        store.shutdown()
        reopened = serving_core.ServingStore(store_dir)
        try:
            assert reopened.get("old-one")["model_id"] == deployment["model_id"]
            assert reopened.events.events()[0]["model_id"] == deployment["model_id"]
        finally:
            reopened.shutdown()
    finally:
        store.shutdown()


def test_loaded_model_sample_hints_use_trained_columns():
    """SimulationEngine examples use a real trained column, not "example"."""
    loaded = {
        "attribute_keys": {
            "proba_dict": ["col_1", "col_2"],
            "columns": ["col_1", "col_2"],
        }
    }
    fields = [
        {
            "name": "df",
            "type": "DataFrame",
            "sample": [{"col_1": 1.0, "col_2": 2.0}],
        },
        {"name": "column", "type": "str", "sample": "example"},
    ]
    hinted = serving_core.apply_sample_hints(fields, loaded)
    assert hinted[1]["sample"] == "col_1"
    assert set(hinted[0]["sample"][0]) == {"col_1", "col_2"}
    payload = serving_core.method_example(hinted)
    assert payload["inputs"]["column"] == "col_1"
    assert payload["inputs"]["df"][0]["col_1"] == 1.0


def test_list_models_hides_stale_and_historical_versions(tmp_path, monkeypatch):
    """The instance picker only shows current, available versions."""
    monkeypatch.setattr(core, "LOAD_RESTRICT", False)
    store = serving_core.ServingStore(tmp_path / "store")
    try:
        valid = tmp_path / "model.pkl"
        valid.write_bytes(pickle.dumps({"ok": True}))
        missing = tmp_path / "missing.pkl"

        def add(model_id, path, created, name="Calculator · predict"):
            record = store._normalise_model_record(
                {
                    "model_id": model_id,
                    "model_type_id": "Calculator",
                    "model_name": "Calculator",
                    "name": name,
                    "parameters": {"value": 1},
                    "load": {"enabled": True, "loader": "load", "path": str(path)},
                    "created_at": created,
                    "updated_at": created,
                }
            )
            store._models[model_id] = record

        add("mdl-old", valid, "2026-01-01T00:00:00Z")
        add("mdl-new", valid, "2026-01-02T00:00:00Z")
        add("mdl-missing", missing, "2026-01-03T00:00:00Z")
        add("mdl-named", valid, "2026-01-04T00:00:00Z", name="Named instance")

        models = store.list_models()
        ids = {model["model_id"] for model in models}
        assert ids == {"mdl-new", "mdl-named"}
        assert all(model["available"] for model in models)
    finally:
        store.shutdown()
