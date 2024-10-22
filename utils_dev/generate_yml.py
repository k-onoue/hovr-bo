import re
import sys
import argparse

def parse_requirements(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    dependencies = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # SSH URLをHTTPSに置き換え
            line = re.sub(r'ssh://git@github\.com/', r'https://github.com/', line)
            dependencies.append(line)
    return dependencies

def get_python_version():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return version

def generate_environment_yml(requirements, python_version, env_name="myenv"):
    environment_yml = f"""
name: {env_name}
dependencies:
  - python={python_version}
  - pip
  - pip:
"""
    for dependency in requirements:
        environment_yml += f"      - {dependency}\n"

    return environment_yml.strip()

def save_to_yml(content, file_path="environment.yml"):
    with open(file_path, 'w') as file:
        file.write(content)
    print(f"environment.yml has been created at {file_path}")

if __name__ == "__main__":
    # argparseの設定
    parser = argparse.ArgumentParser(description="Generate environment.yml from requirements.txt.")
    parser.add_argument("--env_name", type=str, default="hovr-env", help="The name of the conda environment.")
    args = parser.parse_args()

    # ファイルのパス
    requirements_file = "requirements.txt"
    
    # 必要な情報を取得
    requirements = parse_requirements(requirements_file)
    python_version = get_python_version()
    
    # environment.ymlの生成
    environment_yml_content = generate_environment_yml(requirements, python_version=python_version, env_name=args.env_name)
    
    # ファイルに保存
    save_to_yml(environment_yml_content)