import subprocess
import os 

def test_parser_args(model_path, data_path, simulator_name): 
    command = ["diagnose", 
               "--model_path", model_path,
               "--data_path", data_path,
               "--simulator", simulator_name]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 0 

def test_parser_config(config_factory, model_path, data_path, simulator_name): 
    config_path = config_factory(model_path=model_path, data_path=data_path, simulator=simulator_name)
    command = ["diagnose", "--config", config_path]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 0 

def test_main_no_methods(config_factory, model_path, data_path, simulator_name): 
    out_dir = "./test_out_dir/"
    config_path = config_factory(model_path=model_path, data_path=data_path, simulator=simulator_name, plots=[], metrics=[], out_dir=out_dir)
    command = ["diagnose", "--config", config_path]
    process = subprocess.run(command)
    exit_code = process.returncode
    assert exit_code == 0 

    # There should be nothing at the outpath
    assert os.listdir(out_dir) == []

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
