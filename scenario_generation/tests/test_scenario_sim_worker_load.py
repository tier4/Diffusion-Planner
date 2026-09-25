from onnx import TensorProto, helper, save

from scenario_generation import scenario_sim_worker, simulate


def _graph(path, names):
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in names]
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", [names[0]], ["y"])], "g", inputs, [out])
    save(helper.make_model(graph), path)
    return str(path)


def test_an_onnx_is_loaded_by_the_planner_it_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(scenario_sim_worker, "MlPlannerOnnx", lambda p, d: "sampler")
    monkeypatch.setattr(simulate, "load_onnx_model", lambda p, d: ("export", "args"))

    sampler = _graph(tmp_path / "sampler.onnx", ["initial_noise", "lanes"])
    export = _graph(tmp_path / "export.onnx", ["ego_current_state"])

    assert scenario_sim_worker._load(sampler, "cpu") == ("sampler", None)
    assert scenario_sim_worker._load(export, "cpu") == ("export", "args")
