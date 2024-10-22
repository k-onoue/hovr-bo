#!/bin/bash

# ローカルの results ディレクトリと logs ディレクトリを作成
mkdir -p results/
mkdir -p logs/

# 現在の日付を取得（YYYY-MM-DD形式）
DATE="2024-10-23"

# Dockerfile の作成
dockerfile="Dockerfile"

dockerfile_content="
# 基本イメージとして公式の Python イメージを使用
FROM python:3.12

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステム依存ライブラリをインストール
RUN apt-get update && apt-get install -y \\
    pkg-config \\
    libgirepository1.0-dev \\
    gobject-introspection \\
    && rm -rf /var/lib/apt/lists/*

# 必要な Python パッケージをインストール (requirements.txt がある場合)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをすべてコピー
COPY . .

# コンテナ起動時に実行するコマンドを指定
CMD [\"python3\", \"experiments/$DATE/hmc.py\"]
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

# Dockerでコンテナを実行し、resultsディレクトリをボリュームマウント
# ホストのresultsディレクトリをコンテナの/app/resultsにマウント
echo "Dockerコンテナを実行しています..."
docker run --rm \
    -v "$(pwd)/results":/app/results \
    -v "$(pwd)/logs":/app/logs \
    -v "$(pwd)/config.ini":/app/config.ini \
    "$DOCKER_IMAGE"

# 実行完了のメッセージ
echo "Dockerコンテナが終了し、結果はローカルの results ディレクトリに保存されました。"