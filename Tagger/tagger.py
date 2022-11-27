# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np

# given possible tags and ambiguity tags
tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD", "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI", "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0", "UNC", "VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD",
        "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHN", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ", "XX0", "ZZ0", "AJ0-AV0", "AJ0-VVN", "AJ0-VVD", "AJ0-NN1", "AJ0-VVG", "AVP-PRP",  "AVQ-CJS", "CJS-PRP", "CJT-DT0", "CRD-PNI", "NN1-NP0", "NN1-VVB", "NN1-VVG", "NN2-VVZ", "VVD-VVN"]

observations = []

init_prob = {tag: 0 for tag in tags}  # init prob for each tag
transit_prob = {tag: {t: 0 for t in tags}
                for tag in tags}  # transit prob of a tag_y show after tag_x
# emit prob of a word when seeing this tag
emit_prob = {tag: {} for tag in tags}


def viterbi(E, S, I, T, M):
    # row t = time step t = all possible tags for observation t
    # # rows = time step = # observations, # cols = # tags
    prob = np.arange(len(E) * len(S)).reshape(len(E), len(S))
    prev = [[None for _ in range(len(S))] for _ in range(len(E))]

    # determine values for time step 0
    for i in range(len(tags)-1):
        prob[0][i] = init_prob[tags[i]] * emit_prob[tags[i]][observations[0]]
        prev[0][i] = None
    print(prob)

    # find each current state's most likely prior state x
    for t in range(1, len(observations)-1):
        for i in range(len(tags)-1):
            x = max(prob[t-1][x] * transit_prob[x][i]) * \
                emit_prob[i][observations[t]]
            #
            prob[t][i] = prob[t-1][x] * transit_prob[x][i] * \
                emit_prob[i][observations[t]]
            prev[t][i] = x

    return prob, prev

# def create_initial_prob():
#     """ """
# def create_transition_prob():

# def create_emission_prob():


def handle_ambiguity(tag):
    """ return the right order of ambiguity tags, if not, return the original format"""
    if tag == "AV0-AJ0":
        return "AJ0-AV0"
    if tag == "VVN-AJ0":
        return "AJ0-VVN"
    if tag == "VVD-AJ0":
        return "AJ0-VVD"
    if tag == "NN1-AJ0":
        return "AJ0-NN1"
    if tag == "VVG-AJ0":
        return "AJ0-VVG"
    if tag == "PRP-AVP":
        return "AVP-PRP"
    if tag == "CJS-AVQ":
        return "AVQ-CJS"
    if tag == "PRP-CJS":
        return "CJS-PRP"
    if tag == "DT0-CJT":
        return "CJT-DT0"
    if tag == "PNI-CRD":
        return "CRD-PNI"
    if tag == "NP0-NN1":
        return "NN1-NP0"
    if tag == "VVB-NN1":
        return "NN1-VVB"
    if tag == "VVG-NN1":
        return "NN1-VVG"
    if tag == "VVZ-NN2":
        return "NN2-VVZ"
    if tag == "VVN-VVD":
        return "VVD-VVN"

    # if not any of the cases above, should be the right format to insert
    return tag


def read_training_list(training_filename):
    """ read the training file and construct needed model"""
    # count the number of times each tag appears at the beginning of a sentence
    num_sentence = 0

    file = open(training_filename, "r")

    prev_tag = None
    prev_word = None

    for i, pair in enumerate(file):
        pair = pair.split()
        word = pair[0]
        tag = handle_ambiguity(pair[-1])

        if tag not in emit_prob:
            # new tag!
            tags.append(tag)
            emit_prob[tag] = {}
            init_prob[tag] = 1

        if i == 0 or prev_word == ".":
            # the first word and the words after "." are the begining of a new sentence
            num_sentence += 1
            init_prob[tag] += 1

        # count the number of tag showing after prev_tag
        if prev_tag:
            transit_prob[prev_tag][tag] += 1

        if word not in observations:
            observations.append(word)

        # count the number of words given tag
        if word not in emit_prob[tag]:
            emit_prob[tag][word] = 1
        else:  # this word have seen before
            emit_prob[tag][word] += 1

        prev_word = word
        prev_tag = tag

    # now we have a dict in dict with counting, we need to switch it to prob model

    # change the value of init_prob from count to percentage
    for tag_x in init_prob:
        count = init_prob[tag_x]
        percentage = count / num_sentence
        init_prob[tag_x] = percentage

    # change the value of transit_prob from count to percentage
    for tag_x in transit_prob:
        dict = transit_prob[tag_x]

        # sum total number of tags that comes after this tag
        num_tags = 0
        for tag_y in dict:
            num_tags += dict[tag_y]

        for tag_y in tags:
            if tag_y not in dict:  # tag_y never come after tag_x
                count = 0
            else:  # come after
                count = dict[tag_y]

            percentage = count / num_tags
            transit_prob[tag_x][tag_y] = percentage

    # change the value of emit_prob from count to percentage
    for tag_x in emit_prob:
        dict = emit_prob[tag_x]

        # sum total number of words given tag_x
        num_words = 0
        for word_x in dict:
            num_words += dict[word_x]

        for word_x in observations:
            if word_x not in dict:
                count = 0
            else:
                count = dict[word_x]
            percentage = count / num_words
            emit_prob[tag_x][word_x] = percentage

    file.close()


def output_file(output_filename):
    """ print the test output into file """

    file = open(output_filename, "w")

    next_state = Minimax(state)

    file.write(output_format(next_state))

    file.close()


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")

    # count and fill in the global variables on training files
    read_training_list(training_list)

    # run viterbi to create a HMM (hidden = tags, evidence = words)
    viterbi(observations, tags, init_prob, transit_prob, emit_prob)


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    # parameters = sys.argv
    parameters = "py tagger.py -d data/training1.txt -t data/test1.txt -o data/output1.txt"
    training_list = parameters[parameters.index("-d")+3:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
