from dotenv import load_dotenv

load_dotenv()

# app.py
import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================
# 1) 画面設定
# =========================
st.set_page_config(page_title="LangChain LLM Webアプリ", page_icon="🤖", layout="centered")

st.title("🤖 LangChain × LLM かんたんWebアプリ")
st.caption("入力したテキストをLLMに渡して、回答を画面に表示します。専門家の種類もラジオボタンで切り替えできます。")

with st.expander("このWebアプリの概要・操作方法", expanded=True):
    st.markdown(
        """
        - **使い方**
        1. 「専門家の種類」をラジオボタンで選びます（A / B）。
        2. 下の入力フォームに質問や相談内容を入力します。
        3. 「送信」を押すと、LLMの回答が下に表示されます。

        - **ポイント**
        - ラジオボタンの選択に応じて、LLMに渡す **システムメッセージ（役割指示）** が切り替わります。
        - LangChainを使って、入力テキストをプロンプトに組み込み、LLMに投げています。
        """
    )

st.divider()

# =========================
# 2) 専門家（A/B）定義
#    ※「A」「B」は選択肢名として保持しつつ、中身はオリジナルで作成
# =========================
EXPERT_SYSTEM_MESSAGES = {
    "A（Python家庭教師）": (
        "あなたは親切で実践的なPython家庭教師です。"
        "初心者にも分かるように、短い例を交えながら手順を明確に説明してください。"
        "不確かな点は推測せず、確認すべき点を質問してください。"
    ),
    "B（キャリア相談コーチ）": (
        "あなたはキャリア相談のプロのコーチです。"
        "相手の状況を整理し、選択肢を提示し、次の一歩が具体化するように支援してください。"
        "決めつけず、必要に応じて前提確認の質問をしてください。"
    ),
}

# =========================
# 3) LLM呼び出し関数（課題要件）
#    「入力テキスト」と「ラジオボタンでの選択値」を引数で受け、
#    LLMからの回答を戻り値として返す
# =========================
def ask_llm(input_text: str, expert_choice: str) -> str:
    """
    Args:
        input_text: ユーザー入力
        expert_choice: ラジオボタンの選択値（EXPERT_SYSTEM_MESSAGESのキー）
    Returns:
        LLMの回答（文字列）
    """
    system_message = EXPERT_SYSTEM_MESSAGES.get(expert_choice, "あなたは有能なアシスタントです。")

    # Streamlit Community Cloud では st.secrets に OPENAI_API_KEY を入れるのが定石
    # （Settings -> Secrets）
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return "OPENAI_API_KEY が見つかりません。StreamlitのSecretsまたは環境変数に設定してください。"

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": input_text})


# =========================
# 4) UI（ラジオ + 入力フォーム）
# =========================
expert_choice = st.radio(
    "専門家の種類を選んでください（A / B）",
    options=list(EXPERT_SYSTEM_MESSAGES.keys()),
    horizontal=True,
)

with st.form(key="input_form"):
    user_text = st.text_area(
        "入力フォーム（質問・相談など）",
        placeholder="例）Pythonで辞書の使い方を教えて / 転職するか迷っています… など",
        height=140,
    )
    submitted = st.form_submit_button("送信")

if submitted:
    if not user_text.strip():
        st.warning("入力テキストが空です。何か入力してから送信してください。")
    else:
        with st.spinner("LLMに問い合わせ中..."):
            answer = ask_llm(user_text, expert_choice)

        st.subheader("回答結果")
        st.write(answer)

st.divider()
st.caption("※ Streamlit Community Cloud では、Secrets に OPENAI_API_KEY を設定してください。")
