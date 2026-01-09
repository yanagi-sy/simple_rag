"""
RAG統合モジュール

RAGシステム（PDFRagSystem、ReRankingRAG）との統合を管理するモジュール。

このモジュールは、以下の機能を提供します：
- RAGシステムの初期化: PDFRagSystemとReRankingRAGのインスタンスを管理
- 質問応答処理: ユーザーの質問に対して、検索と回答生成を実行（別スレッドで実行）
- プロンプトテンプレートの管理: LLMへの指示文のテンプレートを設定

設計方針：
- 重い処理（検索と回答生成）は別スレッドで実行し、UIがフリーズしないようにする
- RAGシステムの状態を一元管理することで、参照の整合性を保つ
"""

import threading
from datetime import datetime
from typing import Optional

from src.rag import PDFRagSystem, ReRankingRAG
from langchain.prompts import PromptTemplate


class RAGIntegration:
    """
    RAG統合クラス
    
    RAGシステムの初期化、質問応答処理を管理します。
    """
    
    def __init__(self, ui_instance):
        """
        初期化メソッド
        
        Args:
            ui_instance: RAGTerminalUIのインスタンス（他のコンポーネントにアクセスするため）
        """
        self.ui = ui_instance
        
        # RAGシステムのインスタンス変数（初期状態ではNone）
        self.pdf_rag_system: Optional[PDFRagSystem] = None
        self.rag_system: Optional[ReRankingRAG] = None
        
        # プロンプトテンプレートの設定
        # LLMへの指示文のテンプレート（{context}と{question}が後で実際の値に置き換えられる）
        self.prompt_template = PromptTemplate(
            template="""【参考情報】
{context}

【質問】
{question}

【指示】
- 必ず日本語で回答する
- 文脈に書かれている事実のみを使用する
- 推測や一般知識を混ぜない
- 答えられない場合は正直にその旨を伝える
- 余計な説明はせず「結論：〜」の1行だけ出力してください""",
            input_variables=["context", "question"]
        )
    
    def send_message(self, user_input: str):
        """
        ユーザーの質問を処理して回答を生成するメソッド
        
        処理の流れ：
        1. PDFRagSystemが読み込まれているか確認
        2. ReRankingRAGシステムが初期化されていない場合は初期化
        3. 別スレッドで質問応答処理を実行（UIがフリーズしないように）
        4. ハイブリッド検索（Semantic + Keyword）を実行
        5. リランキングで検索結果を再ランキング
        6. LLMで回答を生成
        7. UIに回答を表示
        
        Args:
            user_input: ユーザーの質問
        """
        # PDFRagSystemが読み込まれていない場合
        if self.pdf_rag_system is None:
            self.ui.message_handler.add_error_message(
                "ファイルが読み込まれていません。先にPDFまたはテキストファイルを選択してください。"
            )
            return
        
        # ReRankingRAGシステムが初期化されていない場合は初期化
        if self.rag_system is None:
            try:
                self.rag_system = ReRankingRAG(
                    docs=self.pdf_rag_system.docs,
                    persist_dir=self.pdf_rag_system.persist_dir
                )
                self.rag_system.prompt_template = self.prompt_template
            except Exception as e:
                self.ui.message_handler.add_error_message(f"RAGシステムの初期化に失敗しました: {str(e)}")
                return
        
        # 質問応答処理を別スレッドで実行（UIがフリーズしないように）
        def process_question():
            """
            質問応答処理を実行する内部関数
            
            別スレッドで実行することで、UIを操作し続けられます。
            """
            try:
                # 検索重みを取得（UIから取得）
                w_sem = self.ui.semantic_weight.get()
                w_key = self.ui.keyword_weight.get()
                
                # 検索開始メッセージを表示
                self.ui.root.after(0, lambda: self.ui.message_handler.add_system_message("検索中..."))
                
                # 回答生成開始メッセージを表示
                self.ui.root.after(0, lambda: self.ui.message_handler.add_system_message("回答を生成中..."))
                
                # LLMで回答を生成（内部で検索とプロンプト生成が行われる）
                raw_answer = self.rag_system.answer(
                    question=user_input,
                    k=5,
                    w_sem=w_sem,
                    w_key=w_key,
                    candidate_k=60
                )
                
                # 回答をUIに表示
                self.ui.root.after(0, lambda: self.ui.message_handler.add_assistant_message(raw_answer))
                
                # 会話履歴に追加
                self.ui.conversation_history.append({
                    "role": "user",
                    "message": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                self.ui.conversation_history.append({
                    "role": "assistant",
                    "message": raw_answer,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                # エラーが発生した場合、エラーメッセージを表示
                self.ui.root.after(0, lambda: self.ui.message_handler.add_error_message(
                    f"エラーが発生しました: {str(e)}"
                ))
        
        # 別スレッドを開始
        # daemon=True = メインプログラムが終了したら、このスレッドも終了する
        thread = threading.Thread(target=process_question, daemon=True)
        thread.start()
