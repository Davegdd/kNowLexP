# kNowLexP (knowledge exploration with NLP)
This is a Python module that I developed while conducting my own research about different topics on information sources in natural language. 
It contains several functions and classes that I have found useful to process and extract information mainly from PDF files. 
The module combines Spacy, Haystack and Huggingface models for things such as creating a knowledge graph, extracting related tables and figures, QA, etc.
I intended it to just "get things done" as simply and straightforwardly as possible at the expense of tweaking and minor details. Some examples of what it can do and how below.


## Preprocessing and cleaning

First we install the module and dependencies:

```
pip install knowlexp
pip install farm-haystack
pip install pymupdf

from knowlexp import knlp
```


Then we preprocess the documents by simply passing its path (file or directory) to `doc_loader()`. This function tokenizes the docs and turns them into spacy `Doc` objects
with a custom attribute (doc._.title) which is the name of the file, it also heuristically tries to remove page headers and footers (this last option can be disabled
by passing `clean = False`, not recommended).

 
 ## Knowledge graph
 
 By passing a document created with the previous function to `relations()` as well as any term we are interested in, we can obtain triplets of (subject, root, object)
 that will contain the term. It is important to note that this casts a quite wide (and imprecise) net for the sake of completness and that many results may not be relevant
 (working on making it more accurate). `relations()` will return the triplets as a list of tuples as well as a list of dictionaries that contains the full details
 (triplet, original sentence the triplet was extracted from and document name).
 
 **Example: extracting references to 'infrared' (case insensitive) and its relations from a 67 pages report of a NASA workshop on the search from technological signals from outer space. 
 In the results we can see interesting references to waste heat, M-dwarf stars, dust, infrared excesses, etc**. 
 
https://user-images.githubusercontent.com/108660081/200918907-56e15096-61b8-41e1-ac9e-4460272fd123.mp4

 The previous triplets can already give some information as a raw list but plotting them in a graph may allow to grasp interesting relations. To prepare for plotting,
 we create an object of the class `GraphDF` while passing the triplets, the object will have an attribute, df, that will be a Pandas dataframe of 3 columns (subject, root, object).
 3 methods can be applied to this object: `lemmatize()` which turns the words in the root column into their lemmas ('are' and 'is' become 'be', 'included' and 'includes' become 'include', etc.),
 `subframe()` which will alow us to select the roots we want by passing a list of their indexes or their names and `semantic_grouping()` which returns the roots 
 semantically related to a word (e.g.: if we pass 'observe' it may return 'detect, measure, calculate...'). `lemmatize()` permanently modifies the attribute df and
 returns the dataframe, while `subframe()` and `semantic_grouping()` just return the dataframe without touching the attribute so we can modify the dataframe in palce until
 getting the desired result which then can be given to a new variable.
 
 Once we have a reasonable-sized dataframe with the relations we are interested in we just pass it to `rels_to_graph()` for plotting. The graph will show the name
 of the relation linking two nodes in blue on the edge, we can disable this if we are not interested in showing that information (plotting just one type of relation for example)
 by passing `relations_labels=False`.
 
 **Example: creating a `GraphDF` with the triplets from the term 'infrared', checking the number of each type of relations with value_counts(), increasing this count with
 `lemmatize()`, selecting relations associated to the word 'discovery' using `semantic_grouping()` and plotting**.

https://user-images.githubusercontent.com/108660081/200921233-21c367de-625f-4d59-956f-83c88498da49.mp4


 ## Visual information (tables and figures)
 
 Figures, charts and tables are usually a part very rich in information of any scientific paper. The idea here is not to use image/table extraction but the description
 in the footer of each visual element to find and extract the data we are insterested in, in this way we are leveraging an accurate human-written description to find
 relevant visual information instead of using more expensive image recognition, etc. For this, after creating an object of the class `Visuals` by passing the path 
 (file or directory) to the document/s we want to extract tables and figures from, we can apply 3 methods: `get_descriptions()` (returns and stores in the attribute 
 self.descriptions all the figure/table footers in a list of dictionaries together with some useful metadata), `find_related()` (takes a string as input and returns and
 stores in the attribute self.scores a list ordered by relevance score of the footers related to the inputted string) and `get_pages()` (displays the PDF images themselves
 for visual inspection, they will be displayed in the order found in the attribute self.scores (most relevant first), the number of elements displayed can be limited
 (recommended) by passing top=(integer number of top results to display) or indexes=(list of indexes of the specific figures/tables to display)).
 

## Terminology

A simple feature I have found useful to quickly narrow down the search for relevant documents and to get an idea about the topics they may deal with is to be able to 
easily get a count of specific terms of interest in each document. After considering NER (pure NLP) I discarded it in favour of simple string matching against
a given term list since nothing more than that is needed to gain this relevant insight. The internet in general and Wikipedia in particular are full of tables
of different categories (semiconductor materials, sports teams, proteins, etc.) that can be compared against our documents. Passing the URL of the website the 
table is in to `make_termlist()` and playing around with the argument table_index (defaults to 1, usually the index of a table in a Wikipedia page, for other pages it may be
0 or any other depending on the table we want, we are indexing from all the tables in the website) once we obtain the table we want we can pass its header as a string
to the argument header to get a list of the terms of interest. We can now pass a list of our preprocessed documents and our list of terms to `term_count()` to know which terms appear and
how many times in each document.


## Question Answering and Summarization

Here we leverage huge language models in Huggingface and the integration easiness of the NLP framework Haystack by Deepset to be able to obtain answers directly extracted from
the different documents to questions formulated in natural language, as well as summarization. Here question answering is extractive and summarization abstractive, I 
personally find this approach to be the best given the current state of the art and the use case at hand. To implement this we just create an object of the class
`QAandSummary` passing a list of those of our preprocessed documents we want to get answers from or summarize. The object will use as models "deepset/roberta-base-squad2" for QA
and "facebook/bart-large-xsum" for summarization, these are personal preferences and can be changed by setting the attributes self.reader_model and self.summarizer_model,
respectively. For example, another popular model is Pegasus by Google, we could choose it by doing `our_object_name.summarizer_model = 'google/pegasus-xsum'`, that is,
just passing its name in Huggingface as a string. We can now just make a question with the method `ask()` or summarize each document with the method `summarize()`. We
can also change the documents stored by the object using `overwrite()`, this will eliminate the documents contained and add the new ones passed, we can check the documents
we are getting answers from/summarizing at any time by checking the attribute docs.

Finally, it is worth mentioning the arguments ret_top_k=20 and read_top_k=10 from `ask()` which set the number of document candidates to shortlist for furher analysis
(20) and the maximum number of answers to return (10), respectively, as well as the argument min_length from `summarize()` which sets the minimum length of the summary.
These argument can help to fine-tune and narrow down our answers and summaries depending on the number and length of documents, possible answers, etc.


