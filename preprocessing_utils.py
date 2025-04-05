import re

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
