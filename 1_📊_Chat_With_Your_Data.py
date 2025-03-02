import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_core.tools import BaseTool  # Add this import

from src.logger.base import BaseLogger
from src.models.llms import load_llm
from src.utils import execute_plt_code

# load environment varibles
load_dotenv()
logger = BaseLogger()
MODEL_NAME = "gpt-3.5-turbo"


# Add a new PriceQueryTool
class PriceQueryTool(BaseTool):
    name = "price_query_tool"
    description = "Use this tool when you need to find the price of a specific product."
    
    def _run(self, query: str) -> str:
        """Query the price of a product in the dataframe"""
        df = st.session_state.df
        
        # Check if price column exists
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        if not price_columns:
            return "No price column found in the dataset."
        
        price_col = price_columns[0]
        
        # Check if product name or ID column exists
        product_columns = [col for col in df.columns if 'product' in col.lower() or 'name' in col.lower() or 'item' in col.lower()]
        if not product_columns:
            return "No product name column found in the dataset."
        
        product_col = product_columns[0]
        
        # Try to find the product in the query
        query_lower = query.lower()
        matching_products = df[df[product_col].str.lower().str.contains(query_lower, na=False)]
        
        if matching_products.empty:
            return f"No products found matching '{query}'."
        
        # Format the results
        results = []
        for _, row in matching_products.iterrows():
            product_name = row[product_col]
            price = row[price_col]
            results.append(f"{product_name}: ${price}")
        
        return "\n".join(results)

    async def _arun(self, query: str) -> str:
        """Async implementation of the price query tool"""
        return self._run(query)

def process_query(da_agent, query):

    # Check if it's a price query
    if "price" in query.lower() and "how much" in query.lower():
        price_tool = PriceQueryTool()
        result = price_tool._run(query)
        st.write(result)
        st.session_state.history.append((query, result))
        return

    response = da_agent(query)

    action = response["intermediate_steps"][-1][0].tool_input["query"]

    if "plt" in action:
        st.write(response["output"])

        fig = execute_plt_code(action, df=st.session_state.df)
        if fig:
            st.pyplot(fig)

        st.write("**Executed code:**")
        st.code(action)

        to_display_string = response["output"] + "\n" + f"```python\n{action}\n```"
        st.session_state.history.append((query, to_display_string))

    else:
        st.write(response["output"])
        st.session_state.history.append((query, response["output"]))


def display_chat_history():
    st.markdown("## Chat History: ")
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"**Query: {i+1}:** {q}")
        st.markdown(f"**Response: {i+1}:** {r}")
        st.markdown("---")


def main():

    # Set up streamlit interface
    st.set_page_config(page_title="📊 Smart Data Analysis Tool", page_icon="📊", layout="centered")
    st.header("📊 Smart Data Analysis Tool")
    st.write(
        "### Welcome to our data analysis tool. This tools can assist your daily data analysis tasks. Please enjoy !"
    )

    # Load llms model
    llm = load_llm(model_name=MODEL_NAME)
    logger.info(f"### Successfully loaded {MODEL_NAME} !###")

    # Upload csv file
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here", type="csv")

    # Initial chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Read csv file
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("### Your uploaded data: ", st.session_state.df.head())

        # Create data analysis agent to query with our data
        # Replace line 77-84 with:
        da_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.df,
            agent_type="openai-functions",  # Changed from "tool-calling" to "openai-functions"
            allow_dangerous_code=True,
            verbose=True,
            return_intermediate_steps=True,
        )
        logger.info("### Sucessfully loaded data analysis agent !###")

        # Input query and process query
        query = st.text_input("Enter your questions: ")

        if st.button("Run query"):
            with st.spinner("Processing..."):
                process_query(da_agent, query)

    # Display chat history
    st.divider()
    display_chat_history()


if __name__ == "__main__":
    main()
