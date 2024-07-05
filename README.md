# Description

AI program to help filter the right candidate based on desired skills
from a set of resumes.

The code is based on the post: [blog post](https://archive.ph/1SSDY)

## Usage

Note: the .justfile has common terminal commands you can reuse.
[just app site](https://github.com/casey/just?tab=readme-ov-file#installation)

- add pdf to ./docs directory.  I used Dave Arndt, Chad Upjohn, and Steve Brettschneider resumes.
- need to get ollama running with the llama3 installed
- `poetry` python package for managing virtualenv
- run the `poetry-install` command in the .justfile.  (can simply copy paste to the terminal)
- run one of the calls to the `src/code.py` file.  For instance, `run-data-engr` command.

## Notes

This is just simple app to show how the tools of langchain can fit together so
that summaries of resumes can be used.  Right now, the embeddings are created and stored
in a vector database on each call from the command line.  It does not use an agent
or keeps history, which would be useful for follow on prompts.

## Code Steps

- load docs
- split docs
- add docs to a parent-document-retriever
- set up good prompt
- chain docs into prompt and pass to llm

