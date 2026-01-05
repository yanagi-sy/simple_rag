"""
チャットボットでのプロンプトテンプレート活用例

このスクリプトは、実際のチャットボットアプリケーションで
プロンプトテンプレートをどのように活用するかを示します。

主な機能：
1. 会話履歴の管理
2. 複数のプロンプトテンプレートの使い分け
3. ユーザーコンテキストの管理
4. エラーハンドリング
"""

from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
from datetime import datetime

# =========================================
# チャットボットクラス
# =========================================
class ChatBot:
    """
    プロンプトテンプレートを使ったチャットボットの例
    
    実際のLLMは呼び出しませんが、プロンプトの生成方法を示します
    """
    
    def __init__(self):
        # =========================================
        # プロンプトテンプレートの定義
        # =========================================
        
        # テンプレート1: 通常の会話用
        self.normal_template = PromptTemplate(
            template="""あなたは親切なアシスタントです。

【会話履歴】
{conversation_history}

【ユーザーの質問】
{user_message}

【指示】
- 会話履歴を参考にして、自然な会話をしてください
- 日本語で回答してください
- 簡潔で分かりやすい回答を心がけてください""",
            input_variables=["conversation_history", "user_message"]
        )
        
        # テンプレート2: RAG検索結果を使う場合
        self.rag_template = PromptTemplate(
            template="""あなたは専門知識を持つアシスタントです。

【検索結果（参考情報）】
{search_results}

【会話履歴】
{conversation_history}

【ユーザーの質問】
{user_message}

【指示】
- 検索結果を参考にして、正確な情報を提供してください
- 検索結果にない情報は推測せず、「わかりません」と答えてください
- 日本語で回答してください""",
            input_variables=["search_results", "conversation_history", "user_message"]
        )
        
        # テンプレート3: システム情報を含む場合
        self.system_template = PromptTemplate(
            template="""【システム情報】
現在時刻: {current_time}
ユーザー名: {user_name}
セッションID: {session_id}

【会話履歴】
{conversation_history}

【ユーザーの質問】
{user_message}

【指示】
- システム情報を参考にして、適切な回答をしてください
- ユーザー名を使って親しみやすい会話をしてください
- 日本語で回答してください""",
            input_variables=["current_time", "user_name", "session_id", 
                           "conversation_history", "user_message"]
        )
        
        # テンプレート4: エラーハンドリング用
        self.error_template = PromptTemplate(
            template="""申し訳ございません。エラーが発生しました。

【エラー内容】
{error_message}

【ユーザーの質問】
{user_message}

【指示】
- ユーザーに分かりやすくエラーを説明してください
- 解決方法があれば提案してください
- 日本語で回答してください""",
            input_variables=["error_message", "user_message"]
        )
        
        # 会話履歴を保存するリスト
        self.conversation_history: List[Dict[str, str]] = []
        
        # ユーザー情報
        self.user_name: Optional[str] = None
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def set_user_name(self, name: str):
        """ユーザー名を設定"""
        self.user_name = name
    
    def add_to_history(self, role: str, message: str):
        """会話履歴に追加"""
        self.conversation_history.append({
            "role": role,      # "user" または "assistant"
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def format_conversation_history(self, max_turns: int = 5) -> str:
        """
        会話履歴を文字列にフォーマット
        
        Args:
            max_turns: 表示する会話の最大数（最新のN件）
        """
        if not self.conversation_history:
            return "（会話履歴なし）"
        
        # 最新のmax_turns件だけを取得
        recent_history = self.conversation_history[-max_turns:]
        
        formatted = []
        for entry in recent_history:
            role_label = "ユーザー" if entry["role"] == "user" else "アシスタント"
            formatted.append(f"{role_label}: {entry['message']}")
        
        return "\n".join(formatted)
    
    def generate_prompt_normal(self, user_message: str) -> str:
        """通常の会話用プロンプトを生成"""
        conversation_history = self.format_conversation_history()
        return self.normal_template.format(
            conversation_history=conversation_history,
            user_message=user_message
        )
    
    def generate_prompt_rag(self, user_message: str, search_results: str) -> str:
        """RAG検索結果を使うプロンプトを生成"""
        conversation_history = self.format_conversation_history()
        return self.rag_template.format(
            search_results=search_results,
            conversation_history=conversation_history,
            user_message=user_message
        )
    
    def generate_prompt_system(self, user_message: str) -> str:
        """システム情報を含むプロンプトを生成"""
        conversation_history = self.format_conversation_history()
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        return self.system_template.format(
            current_time=current_time,
            user_name=self.user_name or "ゲスト",
            session_id=self.session_id,
            conversation_history=conversation_history,
            user_message=user_message
        )
    
    def generate_prompt_error(self, user_message: str, error_message: str) -> str:
        """エラーハンドリング用プロンプトを生成"""
        return self.error_template.format(
            error_message=error_message,
            user_message=user_message
        )
    
    def chat(self, user_message: str, mode: str = "normal", 
             search_results: Optional[str] = None) -> str:
        """
        チャット処理のメイン関数
        
        Args:
            user_message: ユーザーのメッセージ
            mode: モード（"normal", "rag", "system"）
            search_results: RAGモードの場合の検索結果
        
        Returns:
            生成されたプロンプト（実際のLLM呼び出しは行わない）
        """
        try:
            # ユーザーのメッセージを履歴に追加
            self.add_to_history("user", user_message)
            
            # モードに応じてプロンプトを生成
            if mode == "rag":
                if not search_results:
                    raise ValueError("RAGモードでは検索結果が必要です")
                prompt = self.generate_prompt_rag(user_message, search_results)
            elif mode == "system":
                prompt = self.generate_prompt_system(user_message)
            else:
                prompt = self.generate_prompt_normal(user_message)
            
            # 実際のLLM呼び出しはここで行う
            # response = self.llm.invoke(prompt)
            # 今回はデモなので、プロンプトを返すだけ
            
            # アシスタントの応答を履歴に追加（デモ用）
            # self.add_to_history("assistant", response)
            
            return prompt
            
        except Exception as e:
            # エラーが発生した場合の処理
            error_prompt = self.generate_prompt_error(user_message, str(e))
            return error_prompt


# =========================================
# 使用例
# =========================================
if __name__ == "__main__":
    print("=" * 80)
    print("チャットボットでのプロンプトテンプレート活用例")
    print("=" * 80)
    
    # チャットボットのインスタンスを作成
    chatbot = ChatBot()
    chatbot.set_user_name("田中太郎")
    
    # =========================================
    # 例1: 通常の会話
    # =========================================
    print("\n【例1】通常の会話モード")
    print("-" * 80)
    
    user_msg1 = "こんにちは"
    prompt1 = chatbot.chat(user_msg1, mode="normal")
    print(f"\nユーザー: {user_msg1}")
    print(f"\n生成されたプロンプト:\n{prompt1}")
    
    # 会話を続ける
    user_msg2 = "今日の天気は？"
    prompt2 = chatbot.chat(user_msg2, mode="normal")
    print(f"\n\nユーザー: {user_msg2}")
    print(f"\n生成されたプロンプト:\n{prompt2}")
    
    # =========================================
    # 例2: RAG検索結果を使う場合
    # =========================================
    print("\n\n" + "=" * 80)
    print("【例2】RAG検索結果を使うモード")
    print("-" * 80)
    
    # 検索結果をシミュレート
    search_results = """
systemctl start httpd  # サービスを起動
systemctl stop httpd   # サービスを停止
systemctl status httpd # サービスの状態を確認
"""
    
    user_msg3 = "systemctlコマンドでサービスを起動する方法を教えてください"
    prompt3 = chatbot.chat(user_msg3, mode="rag", search_results=search_results)
    print(f"\nユーザー: {user_msg3}")
    print(f"\n生成されたプロンプト:\n{prompt3}")
    
    # =========================================
    # 例3: システム情報を含む場合
    # =========================================
    print("\n\n" + "=" * 80)
    print("【例3】システム情報を含むモード")
    print("-" * 80)
    
    user_msg4 = "今何時ですか？"
    prompt4 = chatbot.chat(user_msg4, mode="system")
    print(f"\nユーザー: {user_msg4}")
    print(f"\n生成されたプロンプト:\n{prompt4}")
    
    # =========================================
    # 例4: エラーハンドリング
    # =========================================
    print("\n\n" + "=" * 80)
    print("【例4】エラーハンドリング")
    print("-" * 80)
    
    user_msg5 = "検索結果を教えてください"
    # 意図的に検索結果を渡さない（エラーを発生させる）
    try:
        prompt5 = chatbot.chat(user_msg5, mode="rag")
    except:
        pass
    
    # エラー時のプロンプトを直接生成
    error_prompt = chatbot.generate_prompt_error(
        user_msg5, 
        "RAGモードでは検索結果が必要です"
    )
    print(f"\nユーザー: {user_msg5}")
    print(f"\nエラー時のプロンプト:\n{error_prompt}")
    
    # =========================================
    # 例5: 会話履歴の確認
    # =========================================
    print("\n\n" + "=" * 80)
    print("【例5】会話履歴の確認")
    print("-" * 80)
    
    print(f"\n会話履歴（全{len(chatbot.conversation_history)}件）:")
    for i, entry in enumerate(chatbot.conversation_history, 1):
        role = "ユーザー" if entry["role"] == "user" else "アシスタント"
        print(f"  {i}. [{role}] {entry['message']}")
    
    # =========================================
    # 実用的な活用例の説明
    # =========================================
    print("\n\n" + "=" * 80)
    print("【実用的な活用例】")
    print("=" * 80)
    
    print("""
1. 【マルチモーダル対応】
   - テキスト、画像、音声など、異なる入力タイプに応じて
     異なるプロンプトテンプレートを使い分ける

2. 【ユーザーカスタマイズ】
   - ユーザーの好みや設定に応じて、プロンプトを動的に変更
   - 例：フォーマル/カジュアル、専門的/初心者向け

3. 【コンテキスト管理】
   - 会話履歴、ユーザー情報、システム状態などを
     プロンプトテンプレートに動的に組み込む

4. 【A/Bテスト】
   - 異なるプロンプトテンプレートを試して、
     どちらがより良い結果を生むかテスト

5. 【多言語対応】
   - 言語ごとに異なるプロンプトテンプレートを用意
   - 例：日本語用、英語用、中国語用

6. 【ドメイン特化】
   - 用途に応じてプロンプトを切り替え
   - 例：カスタマーサポート用、技術サポート用、販売用

7. 【セキュリティ】
   - プロンプトインジェクション攻撃を防ぐため、
     ユーザー入力を適切にエスケープしてからテンプレートに埋め込む
""")
    
    print("\n" + "=" * 80)
    print("デモ完了！")
    print("=" * 80)

