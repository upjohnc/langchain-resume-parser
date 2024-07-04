# Description

AI program to help filter the right candidate based on desired skills.
Code based on the post here: [blog post](https://archive.ph/1SSDY)

does not use history - basically the user input is just the desired skills

note: docs need to be added -

-- todo--

- take an input from cli
- figure out how to check that pdf will be read
- add type hints

## Code Steps

- load docs
- split docs
- add docs to a parent-document-retriever
- set up good prompt
- chain docs into prompt and pass to llm

