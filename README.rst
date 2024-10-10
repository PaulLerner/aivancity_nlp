aivancity_nlp
=============

Main repository for the 2024 Natural Language Processing class at
aivancity by Paul Lerner

Classes
-------

- :download:`Introduction to NLP and Distributional Semantics  <./docs/_static/NLP_1_intro_sem.pdf>`

Practical Works
---------------

- Practical Work 1: Distributional Semantics/Skipgram/word2vec https://colab.research.google.com/github/PaulLerner/aivancity_nlp/blob/main/pw1_embedding.ipynb
- Practical Work 2: Transformers https://colab.research.google.com/github/PaulLerner/aivancity_nlp/blob/main/pw2_transformers.ipynb
- Practical Work 3: Large Language Models https://colab.research.google.com/github/PaulLerner/aivancity_nlp/blob/main/pw3_llm.ipynb


Group Homework (50% continous assessment)
-----------------------------------------

**Deadline: Monday 4th of November 23:59 (Paris CEST)** (strict deadline, 5 points malus per day late, so 4 days late means 0/20)

This is a **group work** of **3 members**. You can already start to discuss with your classmates to form groups.

The homework will be an occasion for you to experience groupwork within an NLP project.
In the timeline of the class, you will have learned about Distributional Semantics (word embeddings) and Transformers,
but with the homework you may discover Large Language Models
(before the class on Large Language Models on Tuesday 5th of November).
With this in mind, the topic of the homework will be either:

- "free", if you have an NLP idea that fits with the class, discuss it with me and you might work on it under my guidance
- "default": I will provide a default homework subject, similar to a Practical Work subject but more original and less guided

The homework should not require much more work than a typical Practical Work session (i.e. 4 hours, say max 8 hours).
Therefore, you are not allowed to work on the homework during the classes.
You should be able to start working on it on Thursday 24th of October (so I will not validate "free" topics until then),
i.e. mostly during Toussaint holidays.

You will have to submit your **code** and a **report** which will be graded (instructions below).

Report
^^^^^^

The report should be **a single .pdf file of max. 4 pages** (concision is key).
It should follow the standard IMRAD structure:

- Introduction: present the problem, why it is important. Are there research questions or practical applications?
- Methods: describe the methods you are using to tackle the problem and motivate it:
  why this method and not another?
  What are its advantages and inconvenients?
  What experiment are you running to measure the efficiency or effectiveness of your method to tackle the problem?
- Results: discuss the results of your experiments, either:

  - positive results: your method is efficient or effective to tackle the problem
  - negative results: your method has some caveat (try to find out why, maybe there's a way to solve it in future work)

- Discussion/Conclusion: what did we learn from all of this?
  Try to summarize like:

  - The state of the world before you did this work (that you should have described in the introduction)
  - What your experiments have shown and how.

  You can either end classicaly with the perspectives for this work or abruptly, it's fine.

Remember to add plots and diagrams to illustrate your methods or results if necessary.

Code
^^^^

You can submit your code either as:

- single .zip file with your entire source code (e.g. several .py files)
- link to a GitHub/GitLab repository (in this case, **include the link in your .pdf report**)
- link to a Google Colab Notebook (your code may be quite simple so it may fit in a single notebook;
  likewise, in this case, **include the link in your .pdf report**)


Compute Budget
^^^^^^^^^^^^^^

T4 GPU (on Google Colab) offers 15GB of memory. This should be enough to run inference and fine-tune LLMs of a few billion parameters (or less, obviously)

Note, in `float32`, 1 parameter = 4 bytes so a LLM of 1B parameters holds 4GB of RAM.
But for full fine-tuning, you will need to store gradient activations (without gradient checkpointing) and optimizer states (with optimizers like Adam).

Turn to quantization for cheap inference of larger models or to Parameter Efficient Fine-Tuning for full-fine tuning of LLMs of a few billion parameters.

Much simpler solution: stick to smaller models of hundred of millions of parameters (e.g. BERT, GPT-2, T5).
You're not here to beat the state of the art but to learn NLP.


Contributing
------------

Add Google Colab badges to PWs with https://openincolab.com/

Build docs using `sphinx-build -b html . docs`


Acknowledgements
----------------

This class directly builds upon:

- Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing : An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models (3rd éd.).
- Eisenstein, J. (2019). Natural Language Processing. 587.
- Yejin Choi. (Winter 2024). CSE 447/517: Natural Language Processing (University of Washington Paul G. Allen School of Computer Science & Engineering)
- Noah Smith. (Winter 2023). CSE 447/517: Natural Language Processing (University of Washington Paul G. Allen School of Computer Science & Engineering)
- Benoît Sagot. (2023-2024). Apprendre les langues aux machines (Collège de France)
- Chris Manning. (Spring 2024). Stanford CS224N: Natural Language Processing with Deep Learning
- Classes where I was/am Teacher Assistant:

  - Christopher Kermorvant. Machine Learning for Natural Language Processing (ENSAE)
  - François Landes and Kim Gerdes. Introduction to Machine Learning and NLP (Paris-Saclay)


Also inspired by:

- My PhD thesis: Répondre aux questions visuelles à propos d’entités nommées (2023)
- Noah Smith (2023): Introduction to Sequence Models (LxMLS)
- Kyunghyun Cho: Transformers and Large Pretrained Models (LxMLS 2023), Neural Machine Translation (ALPS 2021)
- My former PhD advisors Olivier Ferret and Camille Guinaudeau and postdoc advisor François Yvon
- My former colleagues at LISN
