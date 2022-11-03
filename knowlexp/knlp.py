import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
import fitz
import os
import re
import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.nodes import TfidfRetriever
from haystack.nodes import TransformersSummarizer
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image, display

try:
    nlp = spacy.load("en_core_web_lg")
except:
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

Doc.set_extension("title", default=None, force=True)


def doc_loader(path, clean=True):
    def cleaner(dir):
        doc = fitz.open(dir)
        headers = []
        footers = []

        for page in doc:
            text = page.get_text()
            headers.append(text.splitlines()[0:5])
            footers.append(text.splitlines()[-5:-1])

        headers = sum(headers, [])
        headers = set([x for x in headers if headers.count(x) > 2 and x.isspace() is False])
        footers = sum(footers, [])
        footers = set([x for x in footers if footers.count(x) > 2 and x.isspace() is False])

        clean_doc = []
        for page in doc:
            text = page.get_text()
            for header in headers:
                text = text.replace(header, "", 1)
            for footer in footers:
                text = text[::-1].replace(footer[::-1], "", 1)[::-1]
            clean_doc.append(text)

        clean_doc = "".join(clean_doc)
        return clean_doc

    docs = []

    if os.path.isdir(path):
        batch = []
        for document in os.listdir(path):
            dir = path + '/' + os.path.basename(document)
            if clean:
                document = cleaner(dir)
            else:
                document = fitz.open(dir)
                document = [page.get_text() for page in document]
                document = "".join(document)
            batch.append((document, {"title": os.path.splitext(os.path.basename(dir))[0]}))

        documents = nlp.pipe(batch, as_tuples=True)
        for doc, title in documents:
            doc._.title = title["title"]
            docs.append(doc)

    elif os.path.isfile(path):
        dir = path
        if clean:
            document = cleaner(dir)
        else:
            document = fitz.open(dir)
            document = [page.get_text() for page in document]
            document = "".join(document)
        document = nlp(document)
        document._.title = os.path.splitext(os.path.basename(dir))[0]
        docs.append(document)

    return docs


def relations(term, text):
    entity_triplets = []
    if type(text) == spacy.tokens.doc.Doc:
        doc = text
    else:
        doc = nlp(text)
    for token in doc:
        if token.text.casefold() == term.casefold():
            root = None

            if token.dep_ == "nsubj" or token.head.dep_ == "nsubj":
                subj = token
                for chunk in token.sent.noun_chunks:
                    if token.i in [token.i for token in chunk]:
                        subj = chunk
                        break

                for word in token.sent:

                    if word.dep_ == "dobj":
                        dobj = word
                        root = dobj.head
                        for chunk in token.sent.noun_chunks:
                            if word.i in [token.i for token in chunk]:
                                dobj = chunk
                        entity_triplets.append(
                            {"Triplet": (subj, root, dobj), "Text": token.sent, "Document": doc._.title})

                    if word.dep_ == "pobj":
                        pobj = word
                        for chunk in token.sent.noun_chunks:
                            if word.i in [token.i for token in chunk]:
                                pobj = chunk
                        for word2 in token.sent:
                            if word2.pos_ in ("VERB", "AUX") and (token.i < word2.i < word.i):
                                root = word2
                                if word2.pos_ == "AUX" and word2.head.pos_ == "VERB":
                                    root = doc[word2.i:word2.head.i + 1]
                                pattern_verb = [{"POS": "AUX"}, {"DEP": "neg", "OP": "?"}, {"POS": "AUX", "OP": "?"},
                                                {"POS": "VERB", "ORTH": word2.text}]
                                pattern_aux = [{"POS": "AUX", "ORTH": word2.text}, {"DEP": "neg", "OP": "?"},
                                               {"POS": "AUX", "OP": "?"}, {"POS": "VERB"}]
                                matcher = Matcher(nlp.vocab)
                                matcher.add("VerbPattern", [pattern_verb, pattern_aux])
                                if matcher(token.sent, as_spans=True):
                                    root = matcher(token.sent, as_spans=True)[0]
                                entity_triplets.append(
                                    {"Triplet": (subj, root, pobj), "Text": token.sent, "Document": doc._.title})
                                break

            if token.dep_ == "pobj" or token.dep_ == "dobj" or token.head.dep_ == "pobj" or token.head.dep_ == "dobj":
                obj = token
                for chunk in token.sent.noun_chunks:
                    if token.i in [token.i for token in chunk]:
                        obj = chunk
                        break

                for word in token.sent:
                    if (word.dep_ == "nsubj" or word.dep_ == "nsubjpass") and word.i < token.i:
                        subj = word
                        if word.pos_ == "PRON" and word.head != word.head.head:
                            subj = word.head.head
                        for chunk in token.sent.noun_chunks:
                            if subj.i in [token.i for token in chunk]:
                                subj = chunk
                                break
                        for word2 in token.sent:
                            if word2.pos_ in ("VERB", "AUX") and (word.i < word2.i < token.i):
                                root = word2
                                pattern_verb = [{"POS": "AUX"}, {"DEP": "neg", "OP": "?"}, {"POS": "AUX", "OP": "?"},
                                                {"POS": "VERB", "ORTH": word2.text}]
                                pattern_aux = [{"POS": "AUX", "ORTH": word2.text}, {"DEP": "neg", "OP": "?"},
                                               {"POS": "AUX", "OP": "?"}, {"POS": "VERB"}]
                                matcher = Matcher(nlp.vocab)
                                matcher.add("VerbPattern", [pattern_verb, pattern_aux])
                                if matcher(token.sent, as_spans=True):
                                    root = matcher(token.sent, as_spans=True)[0]
                        if token.dep_ == "dobj":
                            root = token.head
                        entity_triplets.append(
                            {"Triplet": (subj, root, obj), "Text": token.sent, "Document": doc._.title})

    entity_triplets = [dict(t) for t in {tuple(sorted(d.items())) for d in entity_triplets}]
    return [(dic['Triplet']) for dic in entity_triplets], entity_triplets


class GraphDF:

    def __init__(self, triplets):
        self.df = pd.DataFrame(triplets, columns=['Subj', 'Root', 'Obj'])

    def lemmatize(self):
        lemmatized_roots = []
        for root in self.df['Root']:
            if root is not None:
                lemmatized_roots.append(root.lemma_)
            else:
                lemmatized_roots.append(root)
        lemmatized_roots = pd.Series(lemmatized_roots)
        lemmatized_df = self.df
        lemmatized_df['Root'] = lemmatized_roots.values
        self.df = lemmatized_df

    def subframe(self, relations=[]):
        if all(type(x) is int for x in relations):
            subdf = self.df.iloc[relations]
            return subdf

        elif all(type(x) is str for x in relations):
            subdf = self.df.loc[self.df['Root'].map(str).isin(relations)]
            return subdf

    def semantic_grouping(self, word):
        semantic_group = []
        edges = self.df['Root']
        grouped_roots = edges.value_counts().keys().tolist()
        word = nlp(word)
        for root in grouped_roots:
            if type(root) == str:
                root = nlp(root)
            if root.similarity(word) > 0.49:
                semantic_group.append(root)
        grouped_df = self.df.loc[self.df['Root'].map(str).isin(map(str, semantic_group))]
        return grouped_df


def rels_to_graph(df, relations_labels=True):
    kg_df = pd.DataFrame({'source': df['Subj'], 'edge': df['Root'], 'target': df['Obj']})

    G = nx.from_pandas_edgelist(kg_df, source='source', target='target',
                                edge_attr=True, create_using=nx.MultiDiGraph())

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5)

    if relations_labels:
        nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
        edge_labels = nx.get_edge_attributes(G, 'edge')
        formatted_edge_labels = {(elem[0], elem[1]): edge_labels[elem] for elem in edge_labels}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_color='blue')

    else:
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)

    plt.show()


class Visuals:

    def __init__(self, path=None):
        self.path = path
        self.scores = None
        self.descriptions = None

    def get_descriptions(self):
        path = self.path
        footers = []

        if os.path.isdir(path):
            for paper in os.listdir(path):
                doc = fitz.open(path + '/' + paper)
                pages = [page.get_text('blocks') for page in doc]
                for page in pages:
                    page_number = pages.index(page)
                    for block in page:
                        if re.findall('Figure \d{1,2}\.|Table \d{1,2}\.', block[4][:9]):
                            footers.append(({'Item': block[4][:9], 'Text': block[4][9:], 'Page': page_number + 1,
                                             'Document': os.path.splitext(paper)[0]}))
                        elif re.findall('Fig. \d{1,2}\.', block[4][:9]):
                            footers.append(({'Item': block[4][:7], 'Text': block[4][8:], 'Page': page_number + 1,
                                             'Document': os.path.splitext(paper)[0]}))

        elif os.path.isfile(path):
            doc = fitz.open(path)
            pages = [page.get_text('blocks') for page in doc]
            for page in pages:
                page_number = pages.index(page)
                for block in page:
                    if re.findall('Figure \d{1,2}\.|Table \d{1,2}\.', block[4][:9]):
                        footers.append(({'Item': block[4][:9], 'Text': block[4][9:], 'Page': page_number + 1,
                                         'Document': os.path.splitext(os.path.basename(path))[0]}))
                    elif re.findall('Fig. \d{1,2}\.', block[4][:9]):
                        footers.append(({'Item': block[4][:7], 'Text': block[4][8:], 'Page': page_number + 1,
                                         'Document': os.path.splitext(os.path.basename(path))[0]}))
        self.descriptions = footers
        return self.descriptions

    def find_related(self, text):
        descriptions = self.descriptions
        text = text
        scores = []

        for i in range(len(descriptions)):
            score = nlp(descriptions[i]['Text']).similarity(nlp(text))
            scores.append(('Similarity: ' + str(score), descriptions[i]))
            scores.sort(key=lambda elem: elem[0], reverse=True)

        self.scores = scores
        return self.scores

    def get_pages(self, top=None, indexes=None):
        if self.scores is None:
            scores = self.find_related('')
        else:
            scores = self.scores

        if os.path.isdir(self.path):
            path = self.path

        elif os.path.isfile(self.path):
            path = os.path.split(self.path)[0]

        if indexes is not None:
            for index in indexes:
                file_path = path + '/' + scores[index][1]['Document'] + '.pdf'
                doc = fitz.open(file_path)
                page = doc[scores[index][1]['Page'] - 1]
                pix = page.get_pixmap(dpi=150)
                pix.save("page-%i.png" % page.number)
                display(Image("page-%i.png" % page.number, width=700))
                os.remove("page-%i.png" % page.number)

        else:
            if top is not None:
                top = top

            counter = 0

            for score in scores:
                file_path = path + '/' + score[1]['Document'] + '.pdf'
                doc = fitz.open(file_path)
                page = doc[score[1]['Page'] - 1]
                pix = page.get_pixmap(dpi=150)
                pix.save("page-%i.png" % page.number)
                display(Image("page-%i.png" % page.number, width=700))
                os.remove("page-%i.png" % page.number)
                counter += 1
                if counter == top:
                    break


def make_termlist(url, table_index=1, header=None):
    df = pd.read_html(url, header=0)[table_index]
    if header is None:
        return df
    if header:
        return list(df[str(header)])


def term_count(docs, termlist):
    if type(docs) == list:
        count = []
        for doc in docs:
            for term in termlist:
                occurrences = len(re.findall(r"\b" + str(term) + r"\b", str(doc), re.IGNORECASE))
                if occurrences != 0:
                    count.append((doc._.title, term, occurrences))
        df = pd.DataFrame(count, columns=['Document', 'Term', 'Occurrences'])
        df = df.pivot(index='Document', columns='Term', values='Occurrences')
        return df

    else:
        count = []
        for term in termlist:
            count.append((term, (len(re.findall(r"\b" + str(term) + r"\b", str(docs), re.IGNORECASE)))))
        df = pd.DataFrame(count, columns=['Word', 'Frequency'])
        df.plot(kind='bar', x='Word')


class QAandSummary:
    def __init__(self, documents):
        self._docs = None
        self.document_store = InMemoryDocumentStore()
        self.overwrite(documents)
        self.reader_model = "deepset/roberta-base-squad2"
        self.summarizer_model = "facebook/bart-large-xsum"

    def overwrite(self, documents):
        docs_names = []
        docsforhaystack = []
        for doc in documents:
            docsforhaystack.append({"name": doc._.title, "content": doc.text})
            docs_names.append(doc._.title)
        self.document_store.delete_documents()
        self.document_store.write_documents(docsforhaystack)
        self._docs = docs_names

    @property
    def docs(self):
        return self._docs

    def ask(self, question, ret_top_k=20, read_top_k=10):
        reader = TransformersReader(model_name_or_path=self.reader_model, use_gpu=-1)
        retriever = TfidfRetriever(document_store=self.document_store)
        pipe = ExtractiveQAPipeline(reader, retriever)
        prediction = pipe.run(
            query=question, params={"Retriever": {"top_k": ret_top_k}, "Reader": {"top_k": read_top_k}})
        print_answers(prediction, details="medium")

    def summarize(self, min_length=60, join=False):
        summarizer = TransformersSummarizer(model_name_or_path=self.summarizer_model, min_length=min_length)
        summary = summarizer.predict(documents=self.document_store.get_all_documents(), generate_single_summary=join)
        return [summ.content for summ in summary]