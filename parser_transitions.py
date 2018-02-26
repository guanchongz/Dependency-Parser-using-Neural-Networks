class PartialParse(object):
    def __init__(self, sentence):

        self.sentence = sentence

        self.stack=['ROOT']
        self.buffer=sentence[:]
        self.dependencies=[]


    def parse_step(self, transition):

        if transition=="S":
            self.stack.append(self.buffer.pop(0))
        elif transition=="LA":
            self.dependencies.append((self.stack[-1],self.stack[-2]))
            self.stack.pop(-2)
        else:
            self.dependencies.append((self.stack[-2],self.stack[-1]))
            self.stack.pop(-1)


    def parse(self, transitions):

        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):

    partial_parses = [PartialParse(s) for s in sentences]

    unfinished_parse = partial_parses

    while len(unfinished_parse) > 0:
        minibatch = unfinished_parse[0:batch_size]

        while len(minibatch) > 0:
            transitions = model.predict(minibatch)
            for index, action in enumerate(transitions):
                minibatch[index].parse_step(action)
            minibatch = [parse for parse in minibatch if len(parse.stack) > 1 or len(parse.buffer) > 0]

        unfinished_parse = unfinished_parse[batch_size:]

    dependencies = []
    for n in range(len(sentences)):
        dependencies.append(partial_parses[n].dependencies)


    return dependencies

