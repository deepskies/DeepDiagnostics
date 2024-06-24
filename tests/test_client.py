import subprocess
import os


def test_parser_args(model_path, data_path, simulator_name):
    command = [
        "diagnose",
        "--model_path",
        model_path,
        "--data_path",
        data_path,
        "--simulator",
        simulator_name,
        "--metrics", 
        "", 
        "--plots", 
        ""
    ]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 0



def test_parser_config(config_factory, model_path, data_path, simulator_name):
    config_path = config_factory(
        model_path=model_path, data_path=data_path, simulator=simulator_name, metrics=[], plots=[]
    )
    command = ["diagnose", "--config", config_path]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 0


def test_main_no_methods(config_factory, model_path, data_path, simulator_name, result_output):
    config_path = config_factory(
        model_path=model_path,
        data_path=data_path,
        simulator=simulator_name,
        plots=[],
        metrics=[]
    )
    command = ["diagnose", "--config", config_path]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 0

    # There should be nothing at the outpath
    assert os.listdir(result_output) == []


def test_main_missing_config():
    config_path = "there_is_no_config_at_this_path.yml"
    command = ["diagnose", "--config", config_path]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 1

    
def test_main_missing_args(model_path): 
    command = ["diagnose", "--model_path", model_path]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 1
