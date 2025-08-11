# Podcast 自動パイプライン（Imagine This… → 日本語訳）

- 処理: RSS検知 → 音声DL → 英語文字起こし（OpenAI） → 日本語要約/翻訳/イディオム抽出（ChatGPT） → Assistants API Vector Store に格納
- プロジェクトID: podcast日本語訳
- RSS: https://feeds.captivate.fm/bcg-imagine-this/

## 使い方（GitHub Actions）
1. ファイル一式をリポジトリに作成（このREADME含む）
2. Settings → Secrets and variables → Actions → New repository secret  
   - Name: OPENAI_API_KEY  
   - Value: あなたのAPIキー
3. Actions タブ → 「Podcast Pipeline」→ Run workflow で実行
