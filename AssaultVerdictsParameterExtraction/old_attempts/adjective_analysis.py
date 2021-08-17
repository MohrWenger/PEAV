import conllu

data_file = open("output6.conll",'r', encoding='utf -8')
# sentences = conllu.parse_tree_incr(data)
# for tokentree in conllu.parse_tree_incr(data_file):
#     print(tokentree)

data = "1	גנן	גנן	NN	NN	gen=M|num=S	2	subj\n"+\
"2	גידל	גידל	VB	VB	gen=M|num=S|per=3|tense=PAST	0	ROOT\n"+\
"3	דגן	דגן	NN	NN	gen=M|num=S	2	obj\n"+\
"4	ב	ב	PREPOSITION	PREPOSITION		3	prepmod\n"+\
"5	ה	ה	DEF	DEF		6	def\n"+\
"6	גן	גן	NN	NN	gen=M|num=S	4	pobj\n"+\
"7	.		yyDOT	yyDOT		2	punct\n"
# sentences = conllu.parse_tree_incr(data_file)
# root = sentences[0]
# root.print_tree()
print()
print("children: ")
# print(root.children)
list_of_relevant_roots = []
list_of_relevant_sentences = []
adj_dict = {}
# relevance_flag = False

def BFS (root,checking_func):
    visited = []
    next_to_vis = []
    next_to_vis.append(root)
    relevance_flag = False
    for tok in next_to_vis:
        # if tok.token['upos'] == "JJ":
        #     print("------ you have ecounterdes a JJ -------")
        #     print(tok.token['form']," = ",tok.token['upos'])
        #     print("----------------------------------------")
        if tok.token["deprel"] == "root":
            print(tok.metadata)
        if checking_func(tok):
            print("------ relevant sentnence ------")
            relevance_flag = True
            print("relevance flag is now true, and sentence is ")
            print(tok.token['form'], " = ", tok.token['upos'], "deprel = ")
            print("----------------------------------------")

        else:
            print(tok.token['form']," = ",tok.token['upos'])

        visited.append(tok)
        for child in tok.children:
            if child not in visited:
                next_to_vis.append(child)
    return relevance_flag

def check_if_relevant_noun(tok):
    if tok.token['upos'] == "NN" and tok.token["lemma"] == "מתלונן":

        return True
    else:
        return False
def count_adj_appearence(tok):
    print("upos = ", tok.token['upos'])
    if tok.token['upos'] == "JJ":
        adj_f = tok.token['form']
        if adj_f in adj_dict.keys():
            adj_dict[adj_f] += 1
            if (tok.token["deprel"] == "amod"):
                print (tok.token['form'], "is an amod (:")
            else:
                print("The deprel is: ", tok.token["deprel"])
        else:
            adj_dict[adj_f] = 1
        return True
    else:
        return False


def parse_all_sentences():
    for tokentree in conllu.parse_tree_incr(data_file):

        relevance_flag = BFS(tokentree, check_if_relevant_noun)

        if relevance_flag:
            list_of_relevant_roots.append(tokentree)

    with open('relevant sentences.txt', 'w',encoding="utf-8") as filehandle:
        for listitem in list_of_relevant_roots:
            back_to_conll = listitem.serialize()
            # text = back_to_conll.metadata
            filehandle.write('-----  new sentence  -----\n')
            # filehandle.write('%s\n' % text)
            filehandle.write('%s\n' % back_to_conll)
            filehandle.write('-----  sentence end  -----\n')
    print("relevants = ",list_of_relevant_roots)
    print("num of relevants = ",len(list_of_relevant_roots))


def count_relevant_adj_1(relevant_list):
    for s in relevant_list:
        BFS(s, count_adj_appearence)

parse_all_sentences()
count_relevant_adj_1(list_of_relevant_roots)

sorte = dict(sorted(adj_dict.items(), key=lambda item: item[1]))
print(sorte)
