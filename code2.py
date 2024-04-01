import os
from apikey import apikey
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

os.environ["OPENAI_API_KEY"] = apikey
llm = OpenAI(temperature=0)
st.set_option('deprecation.showPyplotGlobalUse', False)

#Title
st.title("Analyzing Data Powred by AI")

#Keeping the button clicked
if "clicked" not in st.session_state:
    st.session_state.clicked ={1:False}

#button to start the analysis
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Start Full Analysis", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False).reset_index(drop=True)
        for column in df.columns:
            if ":" in column:
                new_name = column.replace(":", "_")
                df.rename(columns={column:new_name}, inplace=True)

        #Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True, handle_parsing_errors=True)

        #EDA Function
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset:")
            st.write(df.head())
            st.write("The shape of your data:")
            st.write(df.shape)
            st.write("Some information about your columns type:")
            st.write(df.dtypes)
            st.write("**Data Explanation**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values and duplicates does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            st.write("**Describing Values in the Data.**")
            st.write(df.describe())
            st.write("**Relationships between variables.**")
            correlation_analysis = pandas_agent.run("Give me a resume about the relationships between variables.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis. In no more then 2 lines.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create? Why?")
            st.write(new_features)
            return
        
        #Category DataViz Function
        @st.cache_data
        def function_dataviz():
            st.header("Visualizations")
            st.subheader("Analyzing Category Columns")
            columns = df.select_dtypes(exclude='number').columns.tolist()
            for column in columns:
                st.write("Value Counts: "+column)
                st.bar_chart(df[column].value_counts())

        #User DataViz Function
        @st.cache_data
        def user_dataviz():
            dataviz_question = pandas_agent.run(user_question_viz + ".Use seaborn or matplotlib and save the figure with the name graph.png. Return just the python code.")
            st.image("graph.png")
            return
        
        #User Question Function
        @st.cache_data
        def user_question_():
            answer = pandas_agent.run(user_question)
            st.write(answer)
            return
        
        #Statistical Recommendations Function
        @st.cache_data
        def statistical_test():
            st.header("Statistical Tests")
            st.write("AI Recommendations")
            stats = pandas_agent.run("What statistical tests to wich column would you recommend to run on this dataset (for example: t-tests, ANOVA, Non-parametric tests, etc.)?")
            st.write(stats)

        #User Statistical Question Function
        def user_question_stats():
            user_stats = st.text_input("Would you like to follow the recommendation? If not, you can request your own model.")
            if user_stats is not None and user_stats!="":
                answer_stats = pandas_agent.run(user_stats)
                st.write(answer_stats)


        #Main App Logic
        function_agent()

        user_question = st.text_input("Would you like to ask something else? If so, write it below.")
        if user_question is not None and user_question!="":
            user_question_()

        function_dataviz()

        user_question_viz = st.text_input("Would you like to create more visualizations? If not, answer: No")
        if user_question_viz is not None and user_question_viz!="":
            user_dataviz()

        statistical_test()
        user_question_stats()
        