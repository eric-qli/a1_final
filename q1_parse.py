# Student name: Eric Li
# Student number: 1007654307
# UTORid: lieric19

'''
This code is provided solely for the personal and private use of students
taking the CSC485H/2501H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Samarendra Dash, Zixin Zhao, Jinman Zhao, Jingcheng Niu, Zhewei Sun

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
'''

"""Functions and classes that handle parsing"""

from itertools import chain

from nltk.parse import DependencyGraph


class PartialParse(object):
    """A PartialParse is a snapshot of an arc-standard dependency parse

    It is fully defined by a quadruple (sentence, stack, next, arcs).

    sentence is a tuple of ordered pairs of (word, tag), where word
    is a a word string and tag is its part-of-speech tag.

    Index 0 of sentence refers to the special "root" node
    (None, self.ROOT_TAG). Index 1 of sentence refers to the sentence's
    first word, index 2 to the second, etc.

    stack is a list of indices referring to elements of
    sentence. The 0-th index of stack should be the bottom of the stack,
    the (-1)-th index is the top of the stack (the side to pop from).

    next is the next index that can be shifted from the buffer to the
    stack. When next == len(sentence), the buffer is empty.

    arcs is a list of triples (idx_head, idx_dep, dependencyRelation) signifying the
    dependency relation `idx_head ->_dependencyRelation idx_dep`, where idx_head is
    the index of the head word, idx_dep is the index of the dependant,
    and dependencyRelation is a string representing the dependency relation label.
    """

    LEFT_ARC_ID = 0
    """An identifier signifying a left arc transition"""

    RIGHT_ARC_ID = 1
    """An identifier signifying a right arc transition"""

    SHIFT_ID = 2
    """An identifier signifying a shift transition"""

    ROOT_TAG = "TOP"
    """A POS-tag given exclusively to the root"""

    def __init__(self, sentence):
        # the initial PartialParse of the arc-standard parse
        # **DO NOT ADD ANY MORE ATTRIBUTES TO THIS OBJECT**
        self.sentence = ((None, self.ROOT_TAG),) + tuple(sentence)
        self.stack = [0]
        self.next = 1
        self.arcs = []

    @property
    def complete(self):
        """bool: return true iff the PartialParse is complete

        Assume that the PartialParse is valid
        """
        return self.next == len(self.sentence) and self.stack == [0]
        

    def parseStep(self, transitionId, dependencyRelation=None):
        """Update the PartialParse with a transition

        Args:
            transitionId : int
                One of LEFT_ARC_ID, RIGHT_ARC_ID, or SHIFT_ID. You
                should check against `self.LEFT_ARC_ID`,
                `self.RIGHT_ARC_ID`, and `self.SHIFT_ID` rather than
                against the values 0, 1, and 2 directly.
            dependencyRelation : str or None
                The dependency label to assign to an arc transition
                (either a left-arc or right-arc). Ignored if
                transitionId == SHIFT_ID

        Raises:
            ValueError if transitionId is an invalid id or is illegal
                given the current state
        """
        if transitionId not in (self.LEFT_ARC_ID, self.RIGHT_ARC_ID, self.SHIFT_ID):
            raise ValueError(f"Invalid transition id: {transitionId}")

        # shift id 
        if transitionId == self.SHIFT_ID:
            if self.next >= len(self.sentence):
                raise ValueError("SHIFT on empty buffer")
            self.stack.append(self.next)
            self.next += 1
            return

        if dependencyRelation is None:
            raise ValueError("Arc transition requires a dependencyRelation label")
        if len(self.stack) < 2:
            raise ValueError("Arc transition requires at least two items on the stack")

        top = self.stack[-1]
        second = self.stack[-2]

        # left arc
        if transitionId == self.LEFT_ARC_ID:
            if second == 0:
                raise ValueError("LEFT-ARC cannot attach ROOT as dependent")
            self.arcs.append((top, second, dependencyRelation))
            self.stack.pop(-2)
            return

        # right arc
        if top == 0:
            raise ValueError("RIGHT-ARC cannot attach ROOT as dependent")
        self.arcs.append((second, top, dependencyRelation))
        self.stack.pop()
        

    def getNLeftMost(self, sentenceIndex, n=None):
        """Returns a list of n leftmost dependants of word

        Leftmost means closest to the beginning of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentenceIndex : refers to word at self.sentence[sentenceIndex]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            dependencyList : The n leftmost dependants as sentence indices.
                If fewer than n, return all dependants. Return in order
                with the leftmost @ 0, immediately right of leftmost @
                1, etc.
        """
        deps = [dep for (head, dep, _) in self.arcs if head == sentenceIndex]
        deps.sort()
        if n is None:
            return deps
        return deps[:n]

    def getNRightMost(self, sentenceIndex, n=None):
        """Returns a list of n rightmost dependants of word on the stack @ idx

        Rightmost means closest to the end of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentenceIndex : refers to word at self.sentence[sentenceIndex]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            dependencyList : The n rightmost dependants as sentence indices. If
                fewer than n, return all dependants. Return in order
                with the rightmost @ 0, immediately left of rightmost @
                1, etc.
        """
        deps = [dep for (head, dep, _) in self.arcs if head == sentenceIndex and dep > sentenceIndex]
        deps.sort(reverse=True)
        if n is None:
            return deps
        return deps[:n]

    def getOracle(self, graph: DependencyGraph):
        """Given a projective dependency graph, determine an appropriate
        transition

        This method chooses either a left-arc, right-arc, or shift so
        that, after repeated calls to pp.parseStep(*pp.getOracle(graph)),
        the arc-transitions this object models matches the
        DependencyGraph "graph". For arcs, it also has to pick out the
        correct dependency relationship.
        graph is projective: informally, this means no crossed lines in the
        dependency graph. More formally, if i -> j and j -> k, then:
             if i > j (left-arc), i > k
             if i < j (right-arc), i < k

        You don't need to worry about API specifics about graph; just call the
        relevant helper functions from the HELPER FUNCTIONS section below. In
        particular, you will (probably) need:
         - getDependencyRelation(i, graph), which will return the dependency relation
           label for the word at index i
         - getHead(i, graph), which will return the index of the head word for
           the word at index i
         - getDependencies(i, graph), which will return the indices of the dependants
           of the word at index i

        Hint: take a look at getLeftDependant and getRightDependant below; their
        implementations may help or give you ideas even if you don't need to
        call the functions themselves.

        *IMPORTANT* if left-arc and shift operations are both valid and
        can lead to the same graph, always choose the left-arc
        operation.

        *ALSO IMPORTANT* make sure to use the values `self.LEFT_ARC_ID`,
        `self.RIGHT_ARC_ID`, `self.SHIFT_ID` for the transition rather than
        0, 1, and 2 directly

        Args:
            graph : nltk.parse.dependencygraph.DependencyGraph
                A projective dependency graph to head towards

        Returns:
            transition, dependencyRelation_label : the next transition to take, along
                with the correct dependency relation label; if transition
                indicates shift, dependencyRelation_label should be None

        Raises:
            ValueError if already completed. Otherwise you can always
            assume that a valid move exists that heads towards the
            target graph
        """
        if self.complete:
            raise ValueError('PartialParse already completed')
        transition, dep_rel_label = -1, None

        if len(self.stack) < 2:
            return self.SHIFT_ID, None

        j = self.stack[-1]
        i = self.stack[-2]

        # Helper
        def attached_deps(idx: int):
            return {dep for (head, dep, _rel) in self.arcs if head == idx}

        def all_gold_deps_attached(idx: int):
            return attached_deps(idx) == set(getDependencies(idx, graph))

        if getHead(i, graph) == j and all_gold_deps_attached(i):
            return self.LEFT_ARC_ID, getDependencyRelation(i, graph)

        if getHead(j, graph) == i and all_gold_deps_attached(j):
            return self.RIGHT_ARC_ID, getDependencyRelation(j, graph)

        return self.SHIFT_ID, None

    def parse(self, transitionDependencyPairs):
        """Applies the provided transitions/dependencyRelations to this PartialParse

        Simply reapplies parseStep for every element in transitionDependencyPairs

        Args:
            transitionDependencyPairs:
                The list of (transitionId, dependencyRelation) pairs in the order
                they should be applied
        Returns:
            The list of arcs produced when parsing the sentence.
            Represented as a list of tuples where each tuple is of
            the form (head, dependent)
        """
        for transitionId, dependencyRelation in transitionDependencyPairs:
            self.parseStep(transitionId, dependencyRelation)
        return self.arcs


def minibatchParse(sentences, model, batchSize):
    """Parses a list of sentences in minibatches using a model.

    Note that parseStep may raise a ValueError if your model predicts an
    illegal (transition, label) pair. Remove any such "stuck" partial-parses
    from the list unfinishedParses.

    Args:
        sentences:
            A list of "sentences", where each element is itself a list
            of pairs of (word, pos)
        model:
            The model that makes parsing decisions. It is assumed to
            have a function model.predict(partialParses) that takes in
            a list of PartialParse as input and returns a list of
            pairs of (transitionId, dependencyRelation) predicted for each parse.
            That is, after calling
                transitionDependencyPairs = model.predict(partialParses)
            transitionDependencyPairs[i] will be the next transition/dependencyRelation pair to apply
            to partialParses[i].
        batchSize:
            The number of PartialParse to include in each minibatch
    Returns:
        arcs:
            A list where each element is the arcs list for a parsed
            sentence. Ordering should be the same as in sentences (i.e.,
            arcs[i] should contain the arcs for sentences[i]).
    """
    partial_parses = [PartialParse(sent) for sent in sentences]
    unfinished = partial_parses[:]

    while unfinished:
        batch = unfinished[:batchSize]

        preds = model.predict(batch)

        stuck = set()
        for pp, (tid, rel) in zip(batch, preds):
            try:
                pp.parseStep(tid, rel)
            except ValueError:
                stuck.add(pp)

        next_unfinished = []
        for pp in unfinished:
            if pp in stuck:
                continue
            if pp.complete:
                continue
            next_unfinished.append(pp)
        unfinished = next_unfinished

    arcs = [pp.arcs for pp in partial_parses]
    return arcs


# *** HELPER FUNCTIONS (look here!) *** #


def getDependencyRelation(sentenceIndex: int, graph: DependencyGraph):
    """Get the dependency relation label for the word at index sentenceIndex
    from the provided DependencyGraph"""
    return graph.nodes[sentenceIndex]['rel']


def getHead(sentenceIndex: int, graph: DependencyGraph):
    """Get the index of the head of the word at index sentenceIndex from the
    provided DependencyGraph"""
    return graph.nodes[sentenceIndex]['head']


def getDependencies(sentenceIndex: int, graph: DependencyGraph):
    """Get the indices of the dependants of the word at index sentenceIndex
    from the provided DependencyGraph"""
    return list(chain(*graph.nodes[sentenceIndex]['deps'].values()))


def getLeftDependant(sentenceIndex: int, graph: DependencyGraph):
    """Get the arc-left dependants of the word at index sentenceIndex from
    the provided DependencyGraph"""
    return (dep for dep in getDependencies(sentenceIndex, graph)
            if dep < graph.nodes[sentenceIndex]['address'])


def getRightDependant(sentenceIndex: int, graph: DependencyGraph):
    """Get the arc-right dependants of the word at index sentenceIndex from
    the provided DependencyGraph"""
    return (dep for dep in getDependencies(sentenceIndex, graph)
            if dep > graph.nodes[sentenceIndex]['address'])


def get_sentence(graph, include_root=False):
    """Get the associated sentence from a DependencyGraph"""
    sentence_w_addresses = [(node['address'], node['word'], node['ctag'])
                            for node in graph.nodes.values()
                            if include_root or node['word'] is not None]
    sentence_w_addresses.sort()
    return tuple(t[1:] for t in sentence_w_addresses)
