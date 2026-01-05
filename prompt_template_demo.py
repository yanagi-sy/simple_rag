"""
プロンプトテンプレートの機能デモンストレーション

このスクリプトは、PromptTemplateがどのように動作するかを示すためのデモです。
実際のLLMを呼び出さずに、プロンプトテンプレートの機能を確認できます。
"""

from langchain.prompts import PromptTemplate

# =========================================
# デモ1: 基本的な使い方
# =========================================
print("=" * 80)
print("【デモ1】基本的な使い方")
print("=" * 80)

# プロンプトテンプレートを作成
# {context} と {question} は変数（プレースホルダー）です
template = """【参考情報】
{context}

【質問】
{question}

【指示】
- 必ず日本語で回答する
- 1つの軸に偏らず複数の評価軸で比較する
- 余計な説明はせず「結論：〜」の1行だけ出力してください"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# テンプレートに実際の値を代入
context_example = """
systemctlは、systemdシステムのサービスを管理するためのコマンドです。
主な機能：
- サービスの起動・停止・再起動
- サービスの状態確認
- サービスの有効化・無効化
"""

question_example = "systemctlコマンドの使い方を教えてください"

# format()メソッドを使って、変数を実際の値に置き換える
formatted_prompt = prompt_template.format(
    context=context_example,
    question=question_example
)

print("\n【テンプレート（変数あり）】")
print(template)
print("\n【実際の値で置き換えた後】")
print(formatted_prompt)

# =========================================
# デモ2: 複数の変数を使う
# =========================================
print("\n" + "=" * 80)
print("【デモ2】複数の変数を使う例")
print("=" * 80)

template2 = """
ユーザー名: {user_name}
質問内容: {question}
回答言語: {language}

上記の質問に{language}で回答してください。
"""

prompt_template2 = PromptTemplate(
    template=template2,
    input_variables=["user_name", "question", "language"]
)

formatted_prompt2 = prompt_template2.format(
    user_name="田中太郎",
    question="Linuxの基本コマンドを教えてください",
    language="日本語"
)

print("\n【テンプレート】")
print(template2)
print("\n【実際の値で置き換えた後】")
print(formatted_prompt2)

# =========================================
# デモ3: 変数の確認
# =========================================
print("\n" + "=" * 80)
print("【デモ3】テンプレートの情報を確認")
print("=" * 80)

print(f"\nテンプレートに含まれる変数: {prompt_template.input_variables}")
print(f"\nテンプレートの内容:")
print(prompt_template.template)

# =========================================
# デモ4: 実際のRAGシステムで使われている例
# =========================================
print("\n" + "=" * 80)
print("【デモ4】実際のRAGシステムでの使用例")
print("=" * 80)

# 検索結果をシミュレート（実際には検索から取得）
search_results = [
    "systemctl start httpd  # サービスを起動",
    "systemctl stop httpd   # サービスを停止",
    "systemctl status httpd # サービスの状態を確認"
]

# 検索結果を1つの文字列に結合（実際のRAGシステムと同じ処理）
context_from_search = "\n\n".join(search_results)
user_question = "systemctlコマンドでサービスを起動する方法は？"

# プロンプトテンプレートを使って、LLMに渡すプロンプトを生成
final_prompt = prompt_template.format(
    context=context_from_search,
    question=user_question
)

print("\n【検索結果（context）】")
print(context_from_search)
print("\n【ユーザーの質問】")
print(user_question)
print("\n【LLMに渡される最終的なプロンプト】")
print(final_prompt)
print("\n※このプロンプトがLLMに渡されると、回答が生成されます")

# =========================================
# デモ5: エラーハンドリング
# =========================================
print("\n" + "=" * 80)
print("【デモ5】エラーハンドリング（変数が不足している場合）")
print("=" * 80)

try:
    # 必要な変数を1つ忘れるとエラーになる
    prompt_template.format(context=context_example)
    # questionが不足しているので、これはエラーになる
except KeyError as e:
    print(f"\nエラーが発生しました: {e}")
    print("→ テンプレートに必要な変数（question）が不足しています")
    print("→ すべての変数を提供する必要があります")

# =========================================
# デモ6: テンプレートの利点
# =========================================
print("\n" + "=" * 80)
print("【デモ6】プロンプトテンプレートを使う利点")
print("=" * 80)

print("""
【利点1】プロンプトの一元管理
- プロンプトを1箇所で管理できる
- 変更が必要な場合、1箇所を修正するだけで済む

【利点2】再利用性
- 同じテンプレートを複数の場所で使い回せる
- コードの重複を避けられる

【利点3】型安全性
- input_variablesで必要な変数を明示できる
- 変数名のタイプミスを防げる

【利点4】可読性
- プロンプトの構造が明確になる
- コードとプロンプトが分離される

【利点5】テストしやすさ
- プロンプトだけをテストできる
- 異なる変数でテストしやすい
""")

print("\n" + "=" * 80)
print("デモ完了！")
print("=" * 80)

