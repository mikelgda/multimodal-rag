import re

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from langchain_community.document_loaders import PyPDFLoader


def load_all_files(root_path, mode="page"):
    docs = []
    for path, folders, files in root_path.walk():
        if len(folders) != 0:
            continue
        for file in files:
            if not file.endswith(".pdf"):
                continue
            doc_loader = PyPDFLoader(path / file, mode=mode, extraction_mode="layout")
            pages = doc_loader.load()
            docs.extend(pages)

    return docs


def preprocess_documents(docs):
    for doc in docs:
        doc.page_content = re.sub(" +", " ", doc.page_content)

    return docs


def load_queries(path):
    with open(path, "r") as f:
        lines = f.readlines()
    queries = []
    for line in lines:
        if line[0] == "#":
            doc_queries = []
        elif line[0] == "@":
            queries.append(doc_queries)
        else:
            doc_queries.append(line.strip())

    return queries


def load_answers(path):
    with open(path, "r") as f:
        lines = f.readlines()

    answers = []
    for line in lines:
        if line[0] == "#":
            doc_answers = []
        elif line[0] == "@":
            answers.append(doc_answers)
        else:
            doc_answers.append(line.strip())

    return answers


def escape_backlashes(text):
    return re.sub(r"\\(?![nt])", r"\\", text)


def create_dataset(rag_chain, queries, answers, delay=70, batch_size=15):
    from time import sleep

    test_cases = []
    rpm = 0
    for doc_queries, doc_answers in zip(queries, answers):
        for query, answer in zip(doc_queries, doc_answers):
            llm_response = rag_chain.invoke({"input": query})["answer"]
            test_case = LLMTestCase(
                input=query,
                expected_output=answer,
                actual_output=escape_backlashes(llm_response),
            )
            rpm += 1
            test_cases.append(test_case)
            if rpm == batch_size:
                print(f"Sleeping after {batch_size} queries")
                sleep(delay)
                rpm = 0
                print("Resuming")
    dataset = EvaluationDataset(test_cases=test_cases)
    return dataset
