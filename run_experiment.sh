#!/bin/bash

# ローカルの results ディレクトリと logs ディレクトリを作成
mkdir -p results/
mkdir -p venv/  # 仮想環境を保存するためのディレクトリを作成

# 現在の日付を取得（YYYY-MM-DD形式）
DATE="2024-10-23"

# Dockerfile の作成
dockerfile="Dockerfile"

dockerfile_content="
# 基本イメージとして公式の Python イメージを使用
FROM python:3.12

# 作業ディレクトリを設定
WORKDIR /app

# 仮想環境のディレクトリをホスト側と共有するボリュームを利用
COPY requirements.txt .  # requirements.txtのみをコピー
RUN python -m venv /app/venv && \\
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをすべてコピー（requirements.txt以外）
COPY . .

# 仮想環境を有効化してからコマンドを実行
CMD [\"/app/venv/bin/python\", \"experiments/$DATE/hmc.py\"]
"

# Dockerfile を書き込み
echo "$dockerfile_content" > $dockerfile

# config.ini ファイルを Docker 内のパスに合わせて上書き
config_file="config.ini"

config_content="[paths]
project_dir = /app
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(results_dir)s/logs"

# config.ini ファイルを上書き
echo "$config_content" > $config_file

# Dockerイメージのビルド
DOCKER_IMAGE="hovr-env"
echo "Dockerイメージをビルドしています..."
docker build -t $DOCKER_IMAGE .

# Dockerでコンテナを実行し、仮想環境をホストに保存して再利用
# ホストのresultsディレクトリ、logsディレクトリ、config.ini、仮想環境をコンテナにマウント
echo "Dockerコンテナを実行しています..."
docker run --rm \
    -v "$(pwd)/results":/app/results \
    -v "$(pwd)/logs":/app/logs \
    -v "$(pwd)/config.ini":/app/config.ini \
    -v "$(pwd)/venv":/app/venv \  # 仮想環境をホスト上に保存
    "$DOCKER_IMAGE"

# 実行完了のメッセージ
echo "Dockerコンテナが終了し、結果はローカルの results ディレクトリに保存されました。"
